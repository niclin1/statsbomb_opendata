"""WP6 — validação externa e **calibração** do modelo final.

Duas partes:

1. **Disponibilidade da Copa do Mundo 2026.** O plano previa validação prospectiva; como a
   Copa terminou em julho/2026, ela passou a ser validação retrospectiva *out-of-sample* —
   ainda sem vazamento, pois o modelo é treinado com dados até 2016 (domínio primário) e o
   torneio é posterior. A primeira tarefa é verificar se a StatsBomb publicou os eventos.
   Se não publicou, o caminho alternativo já documentado é usar os torneios de seleções que
   já estão na base (Copa 2018, Euro 2020, Copa 2022, Euro 2024, Copa América 2024), cujo
   desempenho já foi medido no WP3.

2. **Calibração**, que vale independentemente do item 1 e é conteúdo novo no trabalho. Até
   aqui só se mediu se o modelo *acerta a classe*; calibração mede se as **probabilidades**
   que ele emite correspondem a frequências reais. Um modelo com 46% de acurácia pode ainda
   ser útil se disser "40% de chance de vitória" em situações onde a equipe vence 40% das
   vezes — e inútil se disser 80% nas mesmas situações. Para um caso de uso de apoio à
   decisão no intervalo, a calibração importa tanto quanto a acurácia.

Métricas de calibração reportadas:
    * **Brier score multiclasse** — erro quadrático médio das probabilidades (menor é melhor).
    * **ECE** (Expected Calibration Error) — desvio médio entre confiança e acerto, por faixa.
    * **Curva de confiabilidade** por classe.

Uso:
    python -m src.modeling.wp6_calibracao
"""
from __future__ import annotations

import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from ..common.config import caminho, carregar_config, features
from ..common.run import Run
from ..reporting.tables import escrever_latex
from .evaluate import localizar_run_wp2
from .splits import PARTICOES_INTERNAS, atribuir_particoes, carregar_dataset
from .train import matrizes

N_FAIXAS = 10


def verificar_copa_2026() -> dict:
    """Consulta o catálogo da StatsBomb por uma temporada de Copa do Mundo em 2026."""
    warnings.filterwarnings("ignore")
    try:
        import os
        import requests_cache

        os.makedirs("_cache", exist_ok=True)
        requests_cache.install_cache(os.path.join("_cache", "statsbomb_cache"), backend="sqlite")
        from statsbombpy import sb

        comps = sb.competitions()
        mundiais = comps[comps["competition_name"].str.contains("World Cup", case=False, na=False)]
        temporadas = sorted(mundiais["season_name"].astype(str).unique())
        tem_2026 = any("2026" in s for s in temporadas)
        return {
            "consulta_ok": True,
            "temporadas_de_copa_disponiveis": temporadas,
            "copa_2026_disponivel": tem_2026,
            "competicoes_no_catalogo": int(len(comps)),
        }
    except Exception as erro:
        return {"consulta_ok": False, "erro": f"{type(erro).__name__}: {erro}"}


# ------------------------------------------------------------------- calibração


def brier_multiclasse(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    alvo = np.zeros_like(proba)
    alvo[np.arange(len(y)), y] = 1.0
    return float(((proba - alvo) ** 2).sum(axis=1).mean())


def ece(y: np.ndarray, proba: np.ndarray, n_faixas: int = N_FAIXAS) -> tuple[float, pd.DataFrame]:
    """Expected Calibration Error sobre a classe prevista (confiança máxima)."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    acerto = (pred == y).astype(float)
    bordas = np.linspace(0, 1, n_faixas + 1)
    linhas, erro = [], 0.0
    for i in range(n_faixas):
        m = (conf > bordas[i]) & (conf <= bordas[i + 1])
        if not m.any():
            continue
        c, a, n = conf[m].mean(), acerto[m].mean(), int(m.sum())
        erro += (n / len(y)) * abs(c - a)
        linhas.append({"faixa": f"({bordas[i]:.1f}, {bordas[i+1]:.1f}]", "n": n,
                       "confiança média": round(100 * float(c), 2),
                       "acerto observado": round(100 * float(a), 2),
                       "desvio": round(100 * float(a - c), 2)})
    return float(erro), pd.DataFrame(linhas)


def confiabilidade_por_classe(y: np.ndarray, proba: np.ndarray, classe: int,
                              n_faixas: int = N_FAIXAS) -> pd.DataFrame:
    p = proba[:, classe]
    ocorreu = (y == classe).astype(float)
    bordas = np.linspace(0, 1, n_faixas + 1)
    linhas = []
    for i in range(n_faixas):
        m = (p > bordas[i]) & (p <= bordas[i + 1])
        if m.sum() < 5:
            continue
        linhas.append({"prob_media": float(p[m].mean()),
                       "freq_observada": float(ocorreu[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="WP6 — validação externa e calibração")
    ap.add_argument("--run-wp2", default=None)
    args = ap.parse_args()

    cfg = carregar_config()
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    conjunto = cfg["features"]["conjunto_principal"]
    metrica = cfg["otimizacao"]["metrica_principal"]

    run = Run(cfg, nome="wp6_calibracao")

    print("== 1. disponibilidade dos dados da Copa do Mundo 2026 ==")
    disp = verificar_copa_2026()
    for k, v in disp.items():
        print(f"  {k}: {v}")
    print()

    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    run_wp2 = localizar_run_wp2(cfg, args.run_wp2)
    bst = xgb.Booster()
    bst.load_model(run_wp2 / f"modelo_{conjunto}_{metrica}.ubj")
    cols = features(cfg, conjunto)

    print("== 2. calibração do modelo final, por partição ==")
    particoes = ["teste"] + list(cfg["particao"]["validacao_externa"])
    linhas, detalhes = [], {}
    for nome in particoes:
        sub = df[df["particao"] == nome]
        X, y, _ = matrizes(sub, cfg, conjunto)
        proba = bst.predict(xgb.DMatrix(X[cols]))
        y_arr = y.to_numpy()

        b = brier_multiclasse(y_arr, proba, len(classes))
        e, faixas = ece(y_arr, proba)
        conf_media = float(proba.max(axis=1).mean())
        acerto = float((proba.argmax(axis=1) == y_arr).mean())
        linhas.append({
            "Partição": nome,
            "Partidas": int(sub[cfg["dataset"]["chave_partida"]].nunique()),
            "Brier": round(b, 4),
            "ECE (%)": round(100 * e, 2),
            "Confiança média (%)": round(100 * conf_media, 2),
            "Acerto (%)": round(100 * acerto, 2),
            "Excesso de confiança (p.p.)": round(100 * (conf_media - acerto), 2),
        })
        detalhes[nome] = faixas.to_dict(orient="records")
        print(f"  {nome:22s} Brier {b:.4f} · ECE {100*e:5.2f}% · "
              f"confiança {100*conf_media:5.2f}% vs acerto {100*acerto:5.2f}% "
              f"({100*(conf_media-acerto):+.2f} p.p.)")

    tabela = pd.DataFrame(linhas)

    # ---- curvas de confiabilidade na partição de teste
    sub = df[df["particao"] == "teste"]
    X, y, _ = matrizes(sub, cfg, conjunto)
    proba = bst.predict(xgb.DMatrix(X[cols]))
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ":", color="#6B7773", lw=1, label="calibração perfeita")
    cores = {"Derrota": "#93372A", "Empate": "#9C6317", "Vitória": "#2C6350"}
    for c, nome in sorted(classes.items()):
        curva = confiabilidade_por_classe(y.to_numpy(), proba, c)
        if curva.empty:
            continue
        ax.plot(curva.prob_media, curva.freq_observada, "o-", color=cores[nome], label=nome, ms=4)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Curva de confiabilidade — partição de teste")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    destino = caminho(cfg, "figuras_dir")
    destino.mkdir(parents=True, exist_ok=True)
    for p in (destino / "wp6_confiabilidade.png", run.arquivo("wp6_confiabilidade.png")):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    escrever_latex(
        tabela, caminho(cfg, "tabelas_dir") / "wp6_calibracao.tex",
        legenda=("Calibração do modelo final por partição. \\emph{Brier} é o erro quadrático médio "
                 "das probabilidades (menor é melhor); \\emph{ECE} é o desvio médio entre confiança "
                 "e acerto; \\emph{excesso de confiança} é a diferença entre a confiança média e a "
                 "taxa de acerto — positivo indica um modelo mais confiante do que competente."),
        rotulo="tab:wp6-calibracao",
    )
    tabela.to_csv(run.arquivo("calibracao.csv"), index=False)
    run.escrever_manifest(
        wp6={
            "disponibilidade_copa_2026": disp,
            "modelo": f"modelo_{conjunto}_{metrica}.ubj",
            "run_wp2": run_wp2.name,
            "n_faixas_ece": N_FAIXAS,
            "calibracao": tabela.to_dict(orient="records"),
            "faixas_por_particao": detalhes,
        }
    )
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n" + tabela.to_string(index=False))
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
