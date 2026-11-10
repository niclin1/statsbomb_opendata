"""WP4 — Explicabilidade SHAP das três classes. Contribuição central do TCC II.

O que este módulo produz, e por quê:

1. **SHAP por classe (0=Derrota, 1=Empate, 2=Vitória).** O TCC I fatiava apenas a classe
   Vitória (`train_optuna.py:128`). A pergunta que a pré-banca definiu como foco exige as três.

2. **Teste direto da hipótese da simetria do empate**, feito *sem o modelo*. A hipótese do
   plano é que o empate corresponde a equilíbrio tático entre as equipes — diferenças pequenas
   de xG e Field Tilt. Isso é verificável na base: se for verdade, a distribuição das
   diferenças absolutas por partida é menor nos empates. Testar isso antes do SHAP separa duas
   perguntas que costumam ser confundidas: *o empate é estruturalmente simétrico?* e *o modelo
   percebe essa simetria?* A segunda só faz sentido depois da primeira.

3. **Agregação por bloco temporal** (0–15 / 15–30 / 30–45), para verificar nas três classes a
   afirmação do TCC I de que o bloco 30–45 domina — que lá foi feita olhando só a Vitória.

Onde o SHAP é calculado: na **partição de teste**. É onde o desempenho reportado no WP3 foi
medido, então a explicação e a tabela de resultados descrevem o mesmo comportamento. A
estabilidade do ranqueamento é conferida contra a janela `treino ∪ validação`.

Isto **não** reabre o conjunto de teste: nenhuma métrica de desempenho é recalculada e nenhum
parâmetro do modelo é alterado. É análise descritiva de um modelo já fixado.

Uso:
    python -m src.modeling.explain
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from ..common.config import caminho, carregar_config, features
from ..common.run import Run
from ..reporting.tables import escrever_latex
from .evaluate import localizar_run_wp2
from .splits import atribuir_particoes, carregar_dataset
from .train import janela_otimizacao, matrizes

BLOCOS = ("0_15", "15_30", "30_45")

ROTULOS = {
    "xg_0_15": "xG 0–15", "xg_15_30": "xG 15–30", "xg_30_45": "xG 30–45",
    "ppda_0_15": "PPDA 0–15", "ppda_15_30": "PPDA 15–30", "ppda_30_45": "PPDA 30–45",
    "field_tilt_0_15": "Field Tilt 0–15", "field_tilt_15_30": "Field Tilt 15–30",
    "field_tilt_30_45": "Field Tilt 30–45",
    "team_red_cards_p1": "Cartão vermelho (equipe)",
    "opp_red_cards_p1": "Cartão vermelho (adversário)",
    "is_home": "Manda o jogo",
}


def _estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    })


# --------------------------------------------------------- hipótese da simetria


def analise_simetria(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """A hipótese do empate = equilíbrio tático, testada na base, sem o modelo.

    Para cada partida calcula-se a diferença absoluta entre as duas equipes em cada atributo
    tático, e compara-se a distribuição em partidas empatadas contra decididas. Como a base é
    espelhada (A7), a diferença é obtida diretamente do par de linhas da partida.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    ds = cfg["dataset"]
    chave, alvo = ds["chave_partida"], ds["alvo"]
    p = df.sort_values([chave, "is_home"], ascending=[True, False])
    lado_a = p.iloc[0::2].reset_index(drop=True)
    lado_b = p.iloc[1::2].reset_index(drop=True)
    assert (lado_a[chave].to_numpy() == lado_b[chave].to_numpy()).all()

    empate = (lado_a[alvo] == 1).to_numpy()
    linhas, difs = [], {}
    for col in [c for c in features(cfg, "tatico_11") if not c.startswith("team_red") and not c.startswith("opp_red")]:
        d = (lado_a[col] - lado_b[col]).abs().to_numpy()
        difs[col] = d
        u, p_val = mannwhitneyu(d[empate], d[~empate], alternative="less")
        linhas.append({
            "Atributo": ROTULOS[col],
            "Dif. média — empates": round(float(d[empate].mean()), 4),
            "Dif. média — decididas": round(float(d[~empate].mean()), 4),
            "Razão": round(float(d[empate].mean() / d[~empate].mean()), 3),
            "AUC (dif. → empate)": round(float(roc_auc_score(empate, -d)), 3),
            "p (Mann-Whitney)": f"{p_val:.2e}",
        })

    # separabilidade conjunta: um classificador só com as diferenças absolutas
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_dif = pd.DataFrame(difs)
    auc = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        X_dif, empate, cv=5, scoring="roc_auc",
    )
    resumo = {
        "n_partidas": int(len(lado_a)),
        "n_empates": int(empate.sum()),
        "auc_conjunta_media": round(float(auc.mean()), 4),
        "auc_conjunta_sd": round(float(auc.std(ddof=1)), 4),
    }
    return pd.DataFrame(linhas), resumo


# ------------------------------------------------------------------------ SHAP


def valores_shap(bst: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Devolve sempre no formato (amostras, atributos, classes)."""
    v = shap.TreeExplainer(bst).shap_values(X)
    if isinstance(v, list):
        return np.stack(v, axis=-1)
    return v


def importancia_por_classe(sv: np.ndarray, cols: list[str], classes: dict[int, str]) -> pd.DataFrame:
    linhas = []
    for c, nome in sorted(classes.items()):
        medio = np.abs(sv[:, :, c]).mean(axis=0)
        for col, valor in zip(cols, medio):
            linhas.append({"Classe": nome, "Atributo": ROTULOS[col], "col": col,
                           "SHAP médio |·|": round(float(valor), 5)})
    return pd.DataFrame(linhas)


def tabela_ranqueada(imp: pd.DataFrame, classes: dict[int, str]) -> pd.DataFrame:
    largo = imp.pivot(index="Atributo", columns="Classe", values="SHAP médio |·|")
    largo = largo[[classes[c] for c in sorted(classes)]]
    largo["Média"] = largo.mean(axis=1)
    largo = largo.sort_values("Média", ascending=False).round(4)
    for c in largo.columns:
        largo[f"pos. {c}"] = largo[c].rank(ascending=False).astype(int)
    return largo.reset_index()


def importancia_por_bloco(imp: pd.DataFrame, classes: dict[int, str]) -> pd.DataFrame:
    def bloco(col: str) -> str:
        for b in BLOCOS:
            if col.endswith(b):
                return b.replace("_", "–") + " min"
        return "cartão vermelho"

    imp = imp.assign(Bloco=imp["col"].map(bloco))
    t = imp.groupby(["Bloco", "Classe"])["SHAP médio |·|"].sum().unstack()
    t = t[[classes[c] for c in sorted(classes)]]
    t["Total"] = t.sum(axis=1)
    pct = (100 * t / t.sum()).round(1)
    pct.columns = [f"% {c}" for c in pct.columns]
    return pd.concat([t.round(4), pct], axis=1).reset_index()


# --------------------------------------------------------------------- figuras


def figura_barras(imp: pd.DataFrame, classes: dict[int, str], destino) -> None:
    _estilo()
    ordem = (imp.groupby("Atributo")["SHAP médio |·|"].mean().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(ordem))
    largura = 0.26
    cores = {"Derrota": "#9C3B2E", "Empate": "#9C6317", "Vitória": "#2C6350"}
    for i, c in enumerate(sorted(classes)):
        nome = classes[c]
        vals = [imp[(imp.Classe == nome) & (imp.Atributo == a)]["SHAP médio |·|"].iloc[0] for a in ordem]
        ax.barh(y + (i - 1) * largura, vals, largura, label=nome, color=cores[nome])
    ax.set_yticks(y, ordem)
    ax.set_xlabel("Importância SHAP média |·|")
    ax.set_title("Contribuição dos atributos por classe — partição de teste")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)


def figura_beeswarm(sv: np.ndarray, X: pd.DataFrame, classe: int, nome: str, destino) -> None:
    _estilo()
    plt.figure(figsize=(6.6, 4.2))
    shap.summary_plot(
        sv[:, :, classe], X, feature_names=[ROTULOS[c] for c in X.columns], show=False,
        plot_size=None, color_bar_label="Valor do atributo",
    )
    plt.title(f"SHAP — classe {nome}", fontsize=10)
    plt.xlabel("Valor SHAP (impacto na predição da classe)")
    plt.tight_layout()
    plt.savefig(destino, bbox_inches="tight", dpi=300)
    plt.close()


def figura_dependencia(sv: np.ndarray, X: pd.DataFrame, col: str, classe: int, nome: str, destino) -> None:
    _estilo()
    j = list(X.columns).index(col)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.axhline(0, color="#B9C2BD", lw=0.8, zorder=1)
    ax.scatter(X[col], sv[:, j, classe], s=11, alpha=0.55, color="#9C6317",
               edgecolors="none", zorder=2)
    ax.set_xlabel(ROTULOS[col])
    ax.set_ylabel(f"Valor SHAP — {nome}")
    ax.set_title(f"Dependência: {ROTULOS[col]} → {nome}")
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- execução


def main() -> None:
    ap = argparse.ArgumentParser(description="WP4 — SHAP das três classes")
    ap.add_argument("--particao", default="teste")
    ap.add_argument("--run-wp2", default=None)
    args = ap.parse_args()

    cfg = carregar_config()
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    conjunto = cfg["features"]["conjunto_principal"]
    metrica = cfg["otimizacao"]["metrica_principal"]

    run = Run(cfg, nome="wp4_shap")
    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    run_wp2 = localizar_run_wp2(cfg, args.run_wp2)

    bst = xgb.Booster()
    bst.load_model(run_wp2 / f"modelo_{conjunto}_{metrica}.ubj")
    cols = features(cfg, conjunto)

    figuras = caminho(cfg, "figuras_dir")
    figuras.mkdir(parents=True, exist_ok=True)
    tabelas = caminho(cfg, "tabelas_dir")

    # ---- 1. hipótese da simetria, sem o modelo
    print("== hipótese da simetria do empate (sem o modelo) ==")
    simetria, resumo_sim = analise_simetria(df, cfg)
    print(simetria.to_string(index=False))
    print(f"\n  AUC conjunta (só diferenças absolutas → empate): "
          f"{resumo_sim['auc_conjunta_media']:.4f} (sd {resumo_sim['auc_conjunta_sd']:.4f})"
          f" · {resumo_sim['n_empates']} empates em {resumo_sim['n_partidas']} partidas\n")
    escrever_latex(
        simetria, tabelas / "wp4_simetria_empate.tex",
        legenda=("Diferença absoluta entre as duas equipes por atributo tático, em partidas "
                 "empatadas e decididas (base completa, 2.635 partidas). AUC mede o quanto a "
                 "diferença isolada separa empate de resultado decidido."),
        rotulo="tab:wp4-simetria",
    )

    # ---- 2. SHAP na partição de avaliação
    sub = df[df["particao"] == args.particao]
    X, y, _ = matrizes(sub, cfg, conjunto)
    sv = valores_shap(bst, X)
    imp = importancia_por_classe(sv, cols, classes)
    ranque = tabela_ranqueada(imp, classes)
    blocos = importancia_por_bloco(imp, classes)

    print("== importância SHAP por classe (partição:", args.particao, ") ==")
    print(ranque.to_string(index=False))
    print("\n== agregado por bloco temporal ==")
    print(blocos.to_string(index=False))

    # ---- 3. estabilidade do ranqueamento contra a janela de ajuste
    X_j, _, _ = matrizes(janela_otimizacao(df, cfg), cfg, conjunto)
    imp_j = importancia_por_classe(valores_shap(bst, X_j), cols, classes)
    from scipy.stats import spearmanr
    estabilidade = {}
    for c in sorted(classes):
        nome = classes[c]
        a = imp[imp.Classe == nome].set_index("col")["SHAP médio |·|"]
        b = imp_j[imp_j.Classe == nome].set_index("col")["SHAP médio |·|"].reindex(a.index)
        estabilidade[nome] = round(float(spearmanr(a, b).statistic), 4)
    print("\n== estabilidade do ranqueamento (Spearman, teste × janela de ajuste) ==")
    print(" ", estabilidade)

    # ---- 4. figuras
    figura_barras(imp, classes, figuras / "wp4_shap_barras.png")
    for c, nome in sorted(classes.items()):
        figura_beeswarm(sv, X, c, nome, figuras / f"wp4_shap_beeswarm_{nome.lower()}.png")
    dominante_empate = (
        imp[imp.Classe == classes[1]].sort_values("SHAP médio |·|", ascending=False).iloc[0]["col"]
    )
    figura_dependencia(sv, X, dominante_empate, 1, classes[1],
                       figuras / "wp4_shap_dependencia_empate.png")
    print(f"\n  atributo dominante para o Empate: {ROTULOS[dominante_empate]}")

    # ---- 5. tabelas e manifest
    escrever_latex(
        ranque, tabelas / "wp4_shap_por_classe.tex",
        legenda=("Importância SHAP média absoluta por atributo e classe, calculada na partição de "
                 "teste sobre o modelo final. \\emph{pos.} indica a posição do atributo no "
                 "ranqueamento daquela classe."),
        rotulo="tab:wp4-shap-classe",
    )
    escrever_latex(
        blocos, tabelas / "wp4_shap_por_bloco.tex",
        legenda=("Importância SHAP agregada por bloco temporal do primeiro tempo e por classe."),
        rotulo="tab:wp4-shap-bloco",
    )
    for nome, obj in [("shap_por_classe", ranque), ("shap_por_bloco", blocos), ("simetria_empate", simetria)]:
        obj.to_csv(run.arquivo(f"{nome}.csv"), index=False)

    run.escrever_manifest(
        explicabilidade={
            "particao": args.particao,
            "linhas": int(len(X)),
            "modelo": f"modelo_{conjunto}_{metrica}.ubj",
            "run_wp2": run_wp2.name,
            "atributo_dominante_empate": dominante_empate,
            "estabilidade_spearman": estabilidade,
            "simetria": resumo_sim,
            "importancia_por_classe": ranque.to_dict(orient="records"),
            "importancia_por_bloco": blocos.to_dict(orient="records"),
        }
    )
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
