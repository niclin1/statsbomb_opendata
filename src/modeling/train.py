"""WP2 — treino e otimização honesta.

Corrige os dois defeitos do `src/train_optuna.py` do TCC I:

* **A1**: `n_trials=1`. O "ganho da otimização bayesiana" vinha de um único sorteio.
* **A8**: `xgb.train(param, dtrain)` sem `num_boost_round` → 10 árvores (default da API),
  e o nº de árvores nunca esteve no espaço de busca. O modelo estava subtreinado por
  construção.

E dois problemas de protocolo:

* a busca do TCC I rodava validação cruzada sobre **toda** a base, incluindo o período que
  depois era usado como teste. Aqui a janela de otimização é `treino ∪ validação`, e o
  teste não é tocado (fica para o WP3);
* `TimeSeriesSplit` sobre linhas soltas ignora que cada partida gera **duas** linhas
  espelhadas (A7). As dobras aqui são agrupadas por `match_id`.

A comparação `default × otimizado × baselines` roda sob a mesma validação cruzada — é o
que permite afirmar (ou negar) que a otimização agregou algo.

Uso:
    python -m src.modeling.train                 # estudos completos (config: n_trials)
    python -m src.modeling.train --n-trials 3    # smoke test
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common.config import caminho, carregar_config, features
from ..common.run import Run
from .baselines import metricas
from .splits import atribuir_particoes, carregar_dataset

# Uma função de ajuste recebe (X_treino, y_treino, X_avaliacao) e devolve probabilidades.
Estimador = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


# ------------------------------------------------------------------ dados e dobras


def janela_otimizacao(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """`treino ∪ validação`. O teste não entra — nem aqui, nem no espaço de busca."""
    return df[df["particao"].isin(cfg["otimizacao"]["janela"])]


def matrizes(sub: pd.DataFrame, cfg: dict, conjunto: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    cols = features(cfg, conjunto)
    return sub[cols], sub[cfg["dataset"]["alvo"]], sub[cfg["dataset"]["chave_partida"]].to_numpy()


def dobras(X: pd.DataFrame, y: pd.Series, grupos: np.ndarray, cfg: dict) -> list[tuple]:
    """Dobras estratificadas por classe e **agrupadas por partida**.

    Agrupar é obrigatório (A7): as duas linhas de uma partida têm alvos complementares e
    `field_tilt` somando 1, então separá-las entregaria o alvo da dobra de validação.
    Estratificar mantém a proporção de empates estável entre dobras — sem isso, com 24,6%
    de empates, o F1 macro oscila por sorteio de dobra e não por hiperparâmetro.
    """
    val = cfg["otimizacao"]["validacao"]
    divisor = StratifiedGroupKFold(
        n_splits=val["n_splits"], shuffle=True, random_state=cfg["seed"]
    )
    return list(divisor.split(X, y, groups=grupos))


# --------------------------------------------------------------------- estimadores


def pesos(y: pd.Series, modo: str) -> np.ndarray | None:
    """`balanceado`: peso inversamente proporcional à frequência da classe no treino."""
    if modo == "nenhum":
        return None
    freq = y.value_counts()
    fator = len(y) / (len(freq) * freq)
    return y.map(fator).to_numpy()


def params_xgb(cfg: dict, ajustaveis: dict) -> tuple[dict, int, str]:
    """Separa o que vai para `xgb.train` do que é argumento da chamada."""
    ajustaveis = dict(ajustaveis)
    num_boost_round = int(ajustaveis.pop("num_boost_round"))
    balanceamento = ajustaveis.pop("balanceamento", "nenhum")
    mod = cfg["modelo"]
    params = {
        "objective": mod["objective"],
        "num_class": mod["num_class"],
        "eval_metric": mod["eval_metric"],
        "seed": cfg["seed"],
        "verbosity": 0,
        **ajustaveis,
    }
    return params, num_boost_round, balanceamento


def estimador_xgb(cfg: dict, ajustaveis: dict) -> Estimador:
    params, num_boost_round, balanceamento = params_xgb(cfg, ajustaveis)

    def ajustar(X_tr: pd.DataFrame, y_tr: pd.Series, X_av: pd.DataFrame) -> np.ndarray:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=pesos(y_tr, balanceamento))
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return bst.predict(xgb.DMatrix(X_av))

    return ajustar


def estimador_logistica(cfg: dict) -> Estimador:
    def ajustar(X_tr: pd.DataFrame, y_tr: pd.Series, X_av: pd.DataFrame) -> np.ndarray:
        modelo = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=cfg["seed"]),
        )
        modelo.fit(X_tr, y_tr)
        return modelo.predict_proba(X_av)

    return ajustar


def estimador_classe_majoritaria(n_classes: int) -> Estimador:
    def ajustar(X_tr: pd.DataFrame, y_tr: pd.Series, X_av: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(X_av), n_classes))
        proba[:, int(y_tr.value_counts().idxmax())] = 1.0
        return proba

    return ajustar


def estimador_mando(is_home: pd.Series, n_classes: int) -> Estimador:
    """Regra trivial. Recebe `is_home` por fora porque no conjunto `tatico_11` ela não é
    atributo do modelo, mas continua sendo o baseline a superar (A6)."""

    def ajustar(X_tr: pd.DataFrame, y_tr: pd.Series, X_av: pd.DataFrame) -> np.ndarray:
        casa = is_home.loc[X_av.index].to_numpy()
        proba = np.zeros((len(X_av), n_classes))
        proba[casa == 1, 2] = 1.0
        proba[casa == 0, 0] = 1.0
        return proba

    return ajustar


# ---------------------------------------------------------------------- avaliação


def avaliar_cv(
    estimador: Estimador,
    X: pd.DataFrame,
    y: pd.Series,
    lista_dobras: list[tuple],
    classes: dict[int, str],
    probabilistico: bool = True,
) -> dict:
    """Média e desvio entre dobras. Usa a mesma definição de métrica do WP1/WP3."""
    por_dobra = []
    for i_tr, i_av in lista_dobras:
        X_tr, y_tr = X.iloc[i_tr], y.iloc[i_tr]
        X_av, y_av = X.iloc[i_av], y.iloc[i_av]
        proba = estimador(X_tr, y_tr, X_av)
        m = metricas(y_av, np.argmax(proba, axis=1), classes)
        # Log-Loss só faz sentido para modelos probabilísticos: as regras determinísticas
        # atribuem probabilidade 1 e produziriam um valor arbitrário dependente do clipping.
        m["log_loss"] = (
            round(float(log_loss(y_av, proba, labels=sorted(classes))), 4)
            if probabilistico
            else float("nan")
        )
        por_dobra.append(m)

    tabela = pd.DataFrame(por_dobra)
    return {
        "media": {c: round(float(tabela[c].mean()), 2) for c in tabela.columns},
        "desvio": {c: round(float(tabela[c].std(ddof=1)), 2) for c in tabela.columns},
        "por_dobra": por_dobra,
    }


# ------------------------------------------------------------------------- Optuna


def sugerir(trial: optuna.Trial, espaco: dict) -> dict:
    valores = {}
    for nome, spec in espaco.items():
        if spec["tipo"] == "int":
            valores[nome] = trial.suggest_int(nome, spec["low"], spec["high"])
        elif spec["tipo"] == "float":
            valores[nome] = trial.suggest_float(
                nome, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["tipo"] == "categorico":
            valores[nome] = trial.suggest_categorical(nome, spec["escolhas"])
        else:
            raise ValueError(f"tipo de hiperparâmetro desconhecido: {spec['tipo']}")
    return valores


def rodar_estudo(
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    lista_dobras: list[tuple],
    classes: dict[int, str],
    n_trials: int,
    caminho_db: str | None,
    nome: str,
    metrica: str,
) -> optuna.Study:
    otim = cfg["otimizacao"]

    def objetivo(trial: optuna.Trial) -> float:
        ajustaveis = sugerir(trial, otim["espaco"])
        resultado = avaliar_cv(
            estimador_xgb(cfg, ajustaveis), X, y, lista_dobras, classes
        )
        for chave, valor in resultado["media"].items():
            trial.set_user_attr(chave, valor)
        trial.set_user_attr(f"{metrica}_desvio", resultado["desvio"][metrica])
        return resultado["media"][metrica]

    estudo = optuna.create_study(
        direction=otim["direcao"],
        study_name=nome,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        storage=f"sqlite:///{caminho_db}" if caminho_db else None,
        load_if_exists=False,
    )
    estudo.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    return estudo


# ----------------------------------------------------------------------- execução


def comparar(
    cfg: dict,
    sub: pd.DataFrame,
    conjunto: str,
    n_trials: int,
    run: Run,
) -> tuple[pd.DataFrame, dict[str, optuna.Study], dict[str, dict]]:
    """Baselines (independentes da métrica) + um XGBoost otimizado por métrica declarada."""
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    X, y, grupos = matrizes(sub, cfg, conjunto)
    lista_dobras = dobras(X, y, grupos, cfg)

    def registrar(rotulo: str, otimizado_para: str, estimador: Estimador, probabilistico: bool) -> dict:
        r = avaliar_cv(estimador, X, y, lista_dobras, classes, probabilistico)
        chave = f"{rotulo}_{otimizado_para}".replace(" ", "_").replace("—", "na")
        run.escrever_json(f"cv_{conjunto}_{chave}.json", r)
        return {
            "Atributos": conjunto,
            "Modelo": rotulo,
            "Otimizado para": otimizado_para,
            **{k: r["media"][k] for k in ("acuracia", "f1_macro")},
            "f1_macro_sd": r["desvio"]["f1_macro"],
            **{f"f1_{c}": r["media"][f"f1_{c}"] for c in classes.values()},
            "log_loss": r["media"]["log_loss"],
        }

    linhas = [
        registrar("classe majoritária", "—", estimador_classe_majoritaria(len(classes)), False),
        registrar("mando de campo", "—", estimador_mando(sub["is_home"], len(classes)), False),
        registrar("regressão logística", "—", estimador_logistica(cfg), True),
        registrar("XGBoost default", "—", estimador_xgb(cfg, cfg["modelo"]["params_default"]), True),
    ]

    estudos, melhores_por_metrica = {}, {}
    for metrica in cfg["otimizacao"]["metricas"]:
        estudo = rodar_estudo(
            cfg, X, y, lista_dobras, classes, n_trials,
            caminho_db=run.arquivo(f"study_{conjunto}_{metrica}.db").as_posix(),
            nome=f"xgb_{conjunto}_{metrica}",
            metrica=metrica,
        )
        estudos[metrica] = estudo
        melhores_por_metrica[metrica] = dict(estudo.best_params)
        linhas.append(
            registrar("XGBoost otimizado", metrica, estimador_xgb(cfg, estudo.best_params), True)
        )

    return pd.DataFrame(linhas), estudos, melhores_por_metrica


def main() -> None:
    parser = argparse.ArgumentParser(description="WP2 — otimização honesta")
    parser.add_argument("--n-trials", type=int, default=None, help="sobrepõe otimizacao.n_trials")
    args = parser.parse_args()

    cfg = carregar_config()
    n_trials = args.n_trials or cfg["otimizacao"]["n_trials"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run = Run(cfg, nome="wp2_otimizacao")
    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    sub = janela_otimizacao(df, cfg)

    print(
        f"janela de otimização: {' + '.join(cfg['otimizacao']['janela'])} = "
        f"{sub[cfg['dataset']['chave_partida']].nunique()} partidas / {len(sub)} linhas"
    )
    print(f"n_trials = {n_trials} por conjunto de atributos "
          f"(TCC I: 1) | dobras = {cfg['otimizacao']['validacao']['n_splits']} agrupadas por partida\n")

    tabelas, estudos, melhores_por_conjunto = [], {}, {}
    for conjunto in cfg["otimizacao"]["conjuntos"]:
        print(f"--- {conjunto} ({len(features(cfg, conjunto))} atributos) ---")
        tabela, estudos_conj, melhores = comparar(cfg, sub, conjunto, n_trials, run)
        tabelas.append(tabela)
        estudos[conjunto] = estudos_conj
        melhores_por_conjunto[conjunto] = melhores

        for metrica, estudo in estudos_conj.items():
            pd.DataFrame(
                [
                    {"trial": t.number, "objetivo": t.value, **t.params, **t.user_attrs}
                    for t in estudo.trials
                    if t.value is not None
                ]
            ).to_csv(run.arquivo(f"trials_{conjunto}_{metrica}.csv"), index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(tabela.to_string(index=False), "\n")

    comparativo = pd.concat(tabelas, ignore_index=True)

    # Modelo final de cada conjunto: retreinado em treino ∪ validação inteiras, com os
    # melhores hiperparâmetros. É o que o WP3 vai abrir no teste — uma única vez.
    modelos = {}
    for conjunto, por_metrica in melhores_por_conjunto.items():
        X, y, _ = matrizes(sub, cfg, conjunto)
        for metrica, melhores in por_metrica.items():
            params, num_boost_round, balanceamento = params_xgb(cfg, melhores)
            bst = xgb.train(
                params,
                xgb.DMatrix(X, label=y, weight=pesos(y, balanceamento)),
                num_boost_round=num_boost_round,
            )
            destino = run.arquivo(f"modelo_{conjunto}_{metrica}.ubj")
            bst.save_model(destino)
            modelos[f"{conjunto}/{metrica}"] = str(destino.name)

    from ..reporting.tables import escrever_latex

    rotulos_conjunto = {"tatico_11": "11 (tático)", "completo_12": "12 (+ mando)"}
    para_tex = comparativo.rename(
        columns={
            "Atributos": "Atrib.",
            "acuracia": "Acurácia",
            "f1_macro": "F1 macro",
            "f1_macro_sd": "sd",
            "f1_Derrota": "F1 Derrota",
            "f1_Empate": "F1 Empate",
            "f1_Vitória": "F1 Vitória",
            "log_loss": "Log-Loss",
        }
    ).assign(**{"Atrib.": lambda d: d["Atrib."].map(rotulos_conjunto)})

    escrever_latex(
        para_tex,
        caminho(cfg, "tabelas_dir") / "wp2_comparativo.tex",
        legenda=(
            f"Comparação sob validação cruzada de {cfg['otimizacao']['validacao']['n_splits']} "
            "dobras agrupadas por partida, em treino $\\cup$ validação (o teste não participa). "
            "Valores em \\%, exceto Log-Loss; \\emph{sd} é o desvio-padrão do F1 macro entre "
            "dobras. Log-Loss não se aplica às regras determinísticas."
        ),
        rotulo="tab:wp2-comparativo",
    )
    hiper = pd.DataFrame(
        [
            {
                "Atributos": c,
                "Otimizado para": metrica,
                **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()},
            }
            for c, por_metrica in melhores_por_conjunto.items()
            for metrica, p in por_metrica.items()
        ]
    )
    escrever_latex(
        hiper,
        caminho(cfg, "tabelas_dir") / "wp2_hiperparametros.tex",
        legenda=(
            f"Hiperparâmetros selecionados por {n_trials} tentativas do Optuna (TPE) por "
            "conjunto de atributos, maximizando F1 macro nas dobras de validação."
        ),
        rotulo="tab:wp2-hiperparametros",
    )
    comparativo.to_csv(run.arquivo("comparativo.csv"), index=False)

    importancias, melhores_valores = {}, {}
    for conjunto, por_metrica in estudos.items():
        for metrica, estudo in por_metrica.items():
            chave = f"{conjunto}/{metrica}"
            melhores_valores[chave] = estudo.best_value
            try:
                importancias[chave] = {
                    k: round(float(v), 4)
                    for k, v in optuna.importance.get_param_importances(estudo).items()
                }
            except Exception as erro:  # estudo curto demais (smoke test)
                importancias[chave] = {"erro": str(erro)}

    run.escrever_manifest(
        otimizacao={
            "n_trials": n_trials,
            "janela": cfg["otimizacao"]["janela"],
            "metricas": cfg["otimizacao"]["metricas"],
            "partidas_na_janela": int(sub[cfg["dataset"]["chave_partida"]].nunique()),
            "melhores_params": melhores_por_conjunto,
            "melhor_valor_por_estudo": melhores_valores,
            "importancia_hiperparametros": importancias,
        },
        comparativo=comparativo.to_dict(orient="records"),
        modelos_finais=modelos,
    )

    print("=" * 100)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(comparativo.to_string(index=False))
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
