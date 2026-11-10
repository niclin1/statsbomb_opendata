"""WP3 — avaliação final. O conjunto de teste é aberto UMA ÚNICA VEZ.

Disciplina de execução, e por que ela é técnica e não apenas documental:

* O conjunto de modelos avaliados é **declarado antes** de o teste ser aberto
  (`_planning/resultados_wp2_2026-08-11.md` §7). Escolher o que avaliar depois de ver o
  resultado é a forma mais comum de viés otimista sobreviver a um protocolo correto.
* O código foi depurado rodando sobre a **partição de validação** (`--particao validacao`).
  Nenhuma linha do teste foi lida durante o desenvolvimento.

  Atenção ao interpretar esse ensaio: todos os modelos são ajustados em `treino ∪ validação`,
  logo avaliá-los sobre a validação é medição **dentro da amostra**. No ensaio de 03/09 o
  XGBoost default marcou 100% de acurácia — esse é o resultado *esperado* de um modelo com
  eta 0,3 e 100 árvores reclassificando o próprio treino, e serve como verificação de que os
  modelos foram carregados e estão prevendo. O ensaio valida o encanamento (declaração dos
  modelos, IC bootstrap, tabelas, figura, manifest), **não** os números. Números honestos só
  na partição de teste, que está fora da janela de ajuste.
* Abrir o teste grava `outputs/teste_aberto.json`. Uma segunda execução sobre o teste é
  **recusada**, e só prossegue com `--reabrir "<justificativa>"`, que fica registrada no
  arquivo de guarda e no manifest. O objetivo não é impedir fisicamente — é tornar impossível
  reabrir o teste sem deixar rastro.

Uso:
    python -m src.modeling.evaluate --particao validacao    # ensaio, não toca no teste
    python -m src.modeling.evaluate                         # avaliação final (uma vez)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..common.config import caminho, carregar_config, features, raiz_projeto
from ..common.run import Run
from .baselines import metricas
from .splits import PARTICOES_INTERNAS, atribuir_particoes, carregar_dataset
from .train import (
    estimador_classe_majoritaria,
    estimador_logistica,
    estimador_mando,
    estimador_xgb,
    janela_otimizacao,
    matrizes,
)

ARQUIVO_GUARDA = "outputs/teste_aberto.json"
N_BOOTSTRAP = 2000


# ------------------------------------------------------------------ guarda do teste


def verificar_guarda(particoes: list[str], reabrir: str | None) -> dict | None:
    """Recusa reabrir o teste sem justificativa registrada."""
    if "teste" not in particoes:
        return None
    guarda = raiz_projeto() / ARQUIVO_GUARDA
    if guarda.exists() and not reabrir:
        anterior = json.loads(guarda.read_text(encoding="utf-8"))
        raise SystemExit(
            f"\nO conjunto de teste já foi aberto em {anterior['quando_utc']} "
            f"(run {anterior['run_id']}).\n"
            "Reabrir invalida a estimativa como avaliação única. Se for mesmo necessário, "
            'rode com --reabrir "<justificativa>" — a justificativa fica registrada no arquivo '
            f"de guarda ({ARQUIVO_GUARDA}) e no manifest do run.\n"
        )
    return json.loads(guarda.read_text(encoding="utf-8")) if guarda.exists() else None


def gravar_guarda(run: Run, modelos: list[str], anterior: dict | None, reabrir: str | None) -> None:
    guarda = raiz_projeto() / ARQUIVO_GUARDA
    historico = (anterior or {}).get("historico", [])
    if anterior:
        historico.append({k: anterior[k] for k in ("run_id", "quando_utc", "justificativa_reabertura")})
    guarda.parent.mkdir(parents=True, exist_ok=True)
    guarda.write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "quando_utc": datetime.now(timezone.utc).isoformat(),
                "modelos_declarados": modelos,
                "justificativa_reabertura": reabrir,
                "historico": historico,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------- métricas


def metricas_completas(
    y: pd.Series, proba: np.ndarray, classes: dict[int, str], probabilistico: bool
) -> dict[str, float]:
    from sklearn.metrics import log_loss

    m = metricas(y, np.argmax(proba, axis=1), classes)
    m["log_loss"] = (
        round(float(log_loss(y, proba, labels=sorted(classes))), 4)
        if probabilistico
        else float("nan")
    )
    return m


def ic_bootstrap(
    y: pd.Series,
    proba: np.ndarray,
    grupos: np.ndarray,
    classes: dict[int, str],
    seed: int,
    n: int = N_BOOTSTRAP,
) -> dict[str, tuple[float, float]]:
    """IC 95% por reamostragem **de partidas**, não de linhas.

    As duas linhas de uma partida são dependentes (A7): reamostrar linhas trataria 680
    observações como independentes quando há 340, e estreitaria o intervalo artificialmente.
    Reamostrar partidas preserva a estrutura de pares.
    """
    rng = np.random.default_rng(seed)
    partidas = np.unique(grupos)
    indices_por_partida = {p: np.flatnonzero(grupos == p) for p in partidas}
    pred = np.argmax(proba, axis=1)
    y_arr = y.to_numpy()

    amostras: dict[str, list[float]] = {"acuracia": [], "f1_macro": []}
    for _ in range(n):
        escolhidas = rng.choice(partidas, size=len(partidas), replace=True)
        idx = np.concatenate([indices_por_partida[p] for p in escolhidas])
        m = metricas(pd.Series(y_arr[idx]), pred[idx], classes)
        amostras["acuracia"].append(m["acuracia"])
        amostras["f1_macro"].append(m["f1_macro"])
    return {
        k: (round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2))
        for k, v in amostras.items()
    }


# -------------------------------------------------------------------- estimadores


def localizar_run_wp2(cfg: dict, indicado: str | None) -> Path:
    if indicado:
        p = Path(indicado)
        return p if p.is_absolute() else raiz_projeto() / p
    runs = sorted(caminho(cfg, "runs_dir").glob("*_wp2_otimizacao"))
    if not runs:
        raise SystemExit("nenhum run do WP2 encontrado — rode `python -m src.modeling.train` antes.")
    return runs[-1]


def montar_estimadores(
    cfg: dict, janela: pd.DataFrame, df: pd.DataFrame, run_wp2: Path
) -> list[dict]:
    """Conjunto de modelos DECLARADO antes de o teste ser aberto.

    Os XGBoost otimizados são carregados do run do WP2 (já treinados em treino ∪ validação).
    Regressão logística e XGBoost default são ajustados aqui, sobre a mesma janela, com seed.

    `df` completo é necessário para a regra de mando de campo: ela não é ajustada em nada, mas
    precisa consultar `is_home` das linhas que está prevendo — que ficam fora da janela sempre
    que a partição avaliada não é treino nem validação.
    """
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    manifest = json.loads((run_wp2 / "manifest.json").read_text(encoding="utf-8"))
    principal = cfg["features"]["conjunto_principal"]
    decl: list[dict] = []

    for conjunto in cfg["otimizacao"]["conjuntos"]:
        X_j, y_j, _ = matrizes(janela, cfg, conjunto)
        rotulo_conj = f"{len(features(cfg, conjunto))} atrib."

        for metrica in cfg["otimizacao"]["metricas"]:
            arq = run_wp2 / f"modelo_{conjunto}_{metrica}.ubj"
            bst = xgb.Booster()
            bst.load_model(arq)
            cols = features(cfg, conjunto)
            decl.append(
                {
                    "Modelo": "XGBoost otimizado",
                    "Atributos": rotulo_conj,
                    "Otimizado para": metrica,
                    "principal": conjunto == principal and metrica == cfg["otimizacao"]["metrica_principal"],
                    "probabilistico": True,
                    "conjunto": conjunto,
                    "prever": (lambda b, c: lambda X: b.predict(xgb.DMatrix(X[c])))(bst, cols),
                    "origem": str(arq.relative_to(raiz_projeto())),
                }
            )

        for rotulo, estimador, prob in [
            ("XGBoost default", estimador_xgb(cfg, cfg["modelo"]["params_default"]), True),
            ("regressão logística", estimador_logistica(cfg), True),
            ("classe majoritária", estimador_classe_majoritaria(len(classes)), False),
            ("mando de campo", estimador_mando(df["is_home"], len(classes)), False),
        ]:
            if rotulo in ("classe majoritária", "mando de campo") and conjunto != principal:
                continue  # não dependem de atributos: reportados uma vez só
            decl.append(
                {
                    "Modelo": rotulo,
                    "Atributos": rotulo_conj if rotulo not in ("classe majoritária", "mando de campo") else "—",
                    "Otimizado para": "—",
                    "principal": False,
                    "probabilistico": prob,
                    "conjunto": conjunto,
                    "prever": (lambda e, Xt, yt: lambda X: e(Xt, yt, X))(estimador, X_j, y_j),
                    "origem": "ajustado em treino ∪ validação",
                }
            )
    return decl, manifest


# ------------------------------------------------------------------------ execução


def avaliar(
    cfg: dict, df: pd.DataFrame, decl: list[dict], particoes: list[str], run: Run
) -> pd.DataFrame:
    ds = cfg["dataset"]
    classes = {int(k): v for k, v in ds["classes"].items()}
    linhas = []

    for nome_part in particoes:
        sub = df[df["particao"] == nome_part]
        if sub.empty:
            continue
        for d in decl:
            X, y, grupos = matrizes(sub, cfg, d["conjunto"])
            proba = d["prever"](X)
            m = metricas_completas(y, proba, classes, d["probabilistico"])
            ic = ic_bootstrap(y, proba, grupos, classes, cfg["seed"])
            linhas.append(
                {
                    "Partição": nome_part,
                    "Modelo": d["Modelo"],
                    "Atributos": d["Atributos"],
                    "Otimizado para": d["Otimizado para"],
                    "Partidas": int(sub[ds["chave_partida"]].nunique()),
                    "acuracia": m["acuracia"],
                    "acuracia_ic95": f"[{ic['acuracia'][0]}; {ic['acuracia'][1]}]",
                    "f1_macro": m["f1_macro"],
                    "f1_macro_ic95": f"[{ic['f1_macro'][0]}; {ic['f1_macro'][1]}]",
                    **{f"f1_{c}": m[f"f1_{c}"] for c in classes.values()},
                    "log_loss": m["log_loss"],
                    "principal": d["principal"],
                }
            )
            if d["principal"] and nome_part == particoes[0]:
                _matriz_confusao(y, proba, classes, cfg, run, nome_part)
    return pd.DataFrame(linhas)


def _matriz_confusao(
    y: pd.Series, proba: np.ndarray, classes: dict[int, str], cfg: dict, run: Run, particao: str
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotulos = sorted(classes)
    pred = np.argmax(proba, axis=1)
    conf = np.zeros((len(rotulos), len(rotulos)), dtype=int)
    for i, real in enumerate(rotulos):
        for j, prev in enumerate(rotulos):
            conf[i, j] = int(np.sum((y.to_numpy() == real) & (pred == prev)))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(conf / conf.sum(axis=1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
    nomes = [classes[r] for r in rotulos]
    ax.set_xticks(range(len(rotulos)), nomes)
    ax.set_yticks(range(len(rotulos)), nomes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            pct = 100 * conf[i, j] / conf[i].sum()
            ax.text(j, i, f"{conf[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    color="white" if pct > 50 else "black", fontsize=10)
    ax.set_title(f"Matriz de confusão — modelo principal, partição {particao}", fontsize=10)
    fig.tight_layout()
    destino = caminho(cfg, "figuras_dir")
    destino.mkdir(parents=True, exist_ok=True)
    for p in (destino / f"matriz_confusao_{particao}.png", run.arquivo(f"matriz_confusao_{particao}.png")):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="WP3 — avaliação final")
    ap.add_argument("--particao", default="teste",
                    help="partição interna a avaliar; use 'validacao' para ensaiar sem abrir o teste")
    ap.add_argument("--run-wp2", default=None, help="run do WP2 de onde carregar os modelos")
    ap.add_argument("--sem-externos", action="store_true", help="avalia só a partição interna")
    ap.add_argument("--reabrir", default=None, help="justificativa para reabrir o teste")
    args = ap.parse_args()

    cfg = carregar_config()
    externos = [] if args.sem_externos else list(cfg["particao"]["validacao_externa"])
    particoes = [args.particao] + externos
    anterior = verificar_guarda(particoes, args.reabrir)

    run = Run(cfg, nome=f"wp3_avaliacao_{args.particao}")
    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    janela = janela_otimizacao(df, cfg)
    run_wp2 = localizar_run_wp2(cfg, args.run_wp2)
    decl, manifest_wp2 = montar_estimadores(cfg, janela, df, run_wp2)

    print(f"modelos declarados: {len(decl)} | origem dos otimizados: {run_wp2.name}")
    print(f"partições avaliadas: {', '.join(particoes)}")
    if args.particao == "teste":
        print("*** ABRINDO O CONJUNTO DE TESTE ***" if not anterior else "*** REABRINDO O TESTE ***")

    resultados = avaliar(cfg, df, decl, particoes, run)
    resultados.to_csv(run.arquivo("resultados.csv"), index=False)

    from ..reporting.tables import escrever_latex

    tabelas = caminho(cfg, "tabelas_dir")
    interno = resultados[resultados["Partição"] == args.particao].drop(columns=["Partição", "principal"])

    def para_monografia(t: pd.DataFrame) -> pd.DataFrame:
        """Junta valor e intervalo numa coluna só e usa rótulos legíveis."""
        t = t.copy()
        t["Acurácia (IC 95%)"] = t["acuracia"].map("{:.2f}".format) + " " + t["acuracia_ic95"]
        t["F1 macro (IC 95%)"] = t["f1_macro"].map("{:.2f}".format) + " " + t["f1_macro_ic95"]
        t = t.drop(columns=["acuracia", "acuracia_ic95", "f1_macro", "f1_macro_ic95"]).rename(
            columns={
                "Otimizado para": "Otim. para",
                "f1_Derrota": "F1 Derr.",
                "f1_Empate": "F1 Emp.",
                "f1_Vitória": "F1 Vit.",
                "log_loss": "Log-Loss",
            }
        )
        ordem = [
            "Partição", "Modelo", "Atributos", "Otim. para", "Partidas",
            "Acurácia (IC 95%)", "F1 macro (IC 95%)", "F1 Derr.", "F1 Emp.", "F1 Vit.", "Log-Loss",
        ]
        return t[[c for c in ordem if c in t.columns]]

    escrever_latex(
        para_monografia(interno),
        tabelas / f"wp3_{args.particao}.tex",
        legenda=(
            f"Desempenho na partição {args.particao} ({interno['Partidas'].iloc[0]} partidas). "
            "Valores em \\%, exceto Log-Loss. Intervalos de confiança de 95\\% por reamostragem "
            "\\emph{bootstrap} de partidas (2.000 repetições), que preserva a dependência entre as "
            "duas linhas de cada partida."
        ),
        rotulo=f"tab:wp3-{args.particao}",
    )
    if externos:
        princ = resultados[resultados["principal"] & resultados["Partição"].isin(externos)]
        escrever_latex(
            para_monografia(princ.drop(columns=["principal", "Modelo", "Atributos"])).drop(
                columns=["Otim. para"]
            ),
            tabelas / "wp3_validacao_externa.tex",
            legenda=(
                "Modelo principal nos quatro conjuntos de validação externa. Cada conjunto isola "
                "um deslocamento de distribuição distinto e nenhum participou do treino."
            ),
            rotulo="tab:wp3-externa",
        )

    run.escrever_manifest(
        avaliacao={
            "particao_interna": args.particao,
            "particoes_avaliadas": particoes,
            "run_wp2": run_wp2.name,
            "hiperparametros_wp2": manifest_wp2["otimizacao"]["melhores_params"],
            "modelos_declarados": [
                {k: d[k] for k in ("Modelo", "Atributos", "Otimizado para", "origem")} for d in decl
            ],
            "n_bootstrap": N_BOOTSTRAP,
            "reabertura": args.reabrir,
        },
        resultados=resultados.drop(columns=["principal"]).to_dict(orient="records"),
    )
    if args.particao == "teste":
        gravar_guarda(run, [d["Modelo"] for d in decl], anterior, args.reabrir)

    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print("\n" + interno.to_string(index=False))
        if externos:
            print("\n--- modelo principal na validação externa ---")
            print(
                resultados[resultados["principal"] & resultados["Partição"].isin(externos)][
                    ["Partição", "Partidas", "acuracia", "acuracia_ic95", "f1_macro", "f1_Empate"]
                ].to_string(index=False)
            )
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
