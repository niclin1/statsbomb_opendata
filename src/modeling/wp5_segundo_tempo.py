"""WP5 — o experimento do segundo tempo. Responde à pergunta da banca com número.

A pergunta "por que não usar os dados do segundo tempo?" tem uma resposta óbvia e uma útil.

A óbvia: usar o segundo tempo melhora o desempenho. Isso é esperado **por construção** —
Alberti et al. (2013) reportam que **55% dos gols** de uma partida ocorrem no segundo tempo
(17,7 + 17,2 + 20,2% nos blocos 45–60, 60–75 e 75–90). Um modelo que observa o segundo tempo
não está prevendo o resultado; está observando a maior parte dos eventos que o determinam.

A útil, que é o que este módulo mede: **onde o ganho se concentra.** Se ele crescer de forma
aproximadamente uniforme, o segundo tempo carrega informação tática adicional. Se se concentrar
nos blocos finais, o ganho mede proximidade do apito final — e a exclusão do segundo tempo
deixa de ser uma limitação e passa a ser uma condição do problema: prever *no intervalo*,
quando ainda cabe intervenção tática.

Progressão avaliada (config `features.conjuntos`):
    p5_1t      11 atributos  — apenas 1º tempo (o modelo principal do trabalho)
    p5_ate_60  14            — + bloco 45–60
    p5_ate_75  17            — + bloco 60–75
    p5_ate_90  22            — + bloco 75–90 e cartões vermelhos do 2º tempo

**A comparação roda por validação cruzada sobre `treino ∪ validação`, não sobre o teste.** É uma
pergunta comparativa entre variantes de atributos; respondê-la não exige a estimativa não
enviesada que o teste reservado fornece, e gastá-lo aqui desperdiçaria a única avaliação limpa
disponível. Ressalva a declarar no texto: o WP3 mostrou que ganhos medidos em validação cruzada
não necessariamente transferem para o teste, então as magnitudes abaixo são **teto, não previsão**.

Uso:
    python -m src.modeling.wp5_segundo_tempo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from ..common.config import caminho, carregar_config, features, raiz_projeto
from ..common.run import Run
from ..reporting.tables import escrever_latex
from .splits import atribuir_particoes, carregar_dataset
from .train import (
    avaliar_cv,
    dobras,
    estimador_xgb,
    janela_otimizacao,
    matrizes,
    rodar_estudo,
)

MINUTOS_OBSERVADOS = {"p5_1t": 45, "p5_ate_60": 60, "p5_ate_75": 75, "p5_ate_90": 90}


def localizar_base_v5(cfg: dict) -> Path:
    arqs = sorted(caminho(cfg, "processed_dir").glob("momentum_v5_*.parquet"))
    if not arqs:
        raise SystemExit(
            "base v5 não encontrada — rode `python -m src.etl.momentum_v5` antes."
        )
    return arqs[-1]


def diagnostico_2t(df: pd.DataFrame, cfg: dict) -> dict:
    """Sanidade dos atributos novos antes de modelar."""
    cols_2t = [c for g in ("taticos_45_60", "taticos_60_75", "taticos_75_90", "cartoes_2t")
               for c in cfg["features"][g]]
    sem_2t = int((~df["tem_segundo_tempo"]).sum()) // 2 if "tem_segundo_tempo" in df else 0
    nulos = {c: int(df[c].isna().sum()) for c in cols_2t if df[c].isna().any()}
    zeros_xg = {c: round(100 * float((df[c] == 0).mean()), 2)
                for c in cols_2t if c.startswith("xg_")}
    return {
        "partidas_sem_segundo_tempo": sem_2t,
        "colunas_com_nulos": nulos,
        "xg_zerado_%": zeros_xg,
        "cartao_vermelho_2t_%_linhas": round(
            100 * float((df["team_red_cards_p2"] > 0).mean()), 2
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="WP5 — experimento do segundo tempo")
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args()

    cfg = carregar_config()
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    n_trials = args.n_trials or cfg["wp5"]["n_trials"]
    metrica = cfg["otimizacao"]["metrica_principal"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run = Run(cfg, nome="wp5_segundo_tempo")
    base = localizar_base_v5(cfg)
    df = atribuir_particoes(carregar_dataset(cfg, base), cfg)
    print(f"base: {base.name} | {len(df)} linhas | {df.match_id.nunique()} partidas")
    print("(a partição passou nas verificações de contagem do WP0 — mesmas partidas do v4)\n")

    diag = diagnostico_2t(df, cfg)
    print("== sanidade dos atributos do 2º tempo ==")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print()

    sub = janela_otimizacao(df, cfg)
    print(f"janela: {sub.match_id.nunique()} partidas / {len(sub)} linhas · "
          f"{n_trials} trials por conjunto · métrica {metrica}\n")

    linhas, por_dobra = [], {}
    for conjunto in cfg["wp5"]["conjuntos"]:
        cols = features(cfg, conjunto)
        X, y, grupos = matrizes(sub, cfg, conjunto)
        lista_dobras = dobras(X, y, grupos, cfg)

        estudo = rodar_estudo(
            cfg, X, y, lista_dobras, classes, n_trials,
            caminho_db=run.arquivo(f"study_{conjunto}.db").as_posix(),
            nome=f"xgb_{conjunto}", metrica=metrica,
        )
        r_otim = avaliar_cv(estimador_xgb(cfg, estudo.best_params), X, y, lista_dobras, classes)
        r_dflt = avaliar_cv(
            estimador_xgb(cfg, cfg["modelo"]["params_default"]), X, y, lista_dobras, classes
        )
        por_dobra[conjunto] = r_otim["por_dobra"]
        run.escrever_json(f"cv_{conjunto}.json", {"otimizado": r_otim, "default": r_dflt})

        linhas.append({
            "Conjunto": conjunto,
            "Minutos observados": MINUTOS_OBSERVADOS[conjunto],
            "Atributos": len(cols),
            **{k: r_otim["media"][k] for k in ("acuracia", "f1_macro")},
            **{f"f1_{c}": r_otim["media"][f"f1_{c}"] for c in classes.values()},
            "log_loss": r_otim["media"]["log_loss"],
            "acuracia_default": r_dflt["media"]["acuracia"],
            "f1_macro_default": r_dflt["media"]["f1_macro"],
        })
        print(f"  {conjunto:11s} ({len(cols):2d} atrib., {MINUTOS_OBSERVADOS[conjunto]} min): "
              f"acurácia {r_otim['media']['acuracia']:.2f} · F1 macro {r_otim['media']['f1_macro']:.2f} "
              f"· F1 empate {r_otim['media'][f'f1_{classes[1]}']:.2f}")

    tabela = pd.DataFrame(linhas)

    # ---- ganho incremental, emparelhado nas mesmas dobras
    print("\n== ganho incremental (emparelhado nas mesmas 5 dobras) ==")
    incrementos = []
    base_conj = cfg["wp5"]["conjuntos"][0]
    for i, conjunto in enumerate(cfg["wp5"]["conjuntos"]):
        if i == 0:
            continue
        anterior = cfg["wp5"]["conjuntos"][i - 1]
        a = pd.DataFrame(por_dobra[anterior])
        b = pd.DataFrame(por_dobra[conjunto])
        base_df = pd.DataFrame(por_dobra[base_conj])
        reg = {"De → para": f"{MINUTOS_OBSERVADOS[anterior]} → {MINUTOS_OBSERVADOS[conjunto]} min"}
        for m in ("acuracia", "f1_macro"):
            d_inc = b[m] - a[m]
            d_acum = b[m] - base_df[m]
            reg[f"Δ {m} (bloco)"] = round(float(d_inc.mean()), 2)
            reg[f"dobras {m}"] = f"{int((d_inc > 0).sum())}/5"
            reg[f"Δ {m} (acum. vs 45 min)"] = round(float(d_acum.mean()), 2)
        incrementos.append(reg)
        print(f"  {reg['De → para']:18s} Δacurácia {reg['Δ acuracia (bloco)']:+6.2f} "
              f"({reg['dobras acuracia']}) · Δf1_macro {reg['Δ f1_macro (bloco)']:+6.2f} "
              f"({reg['dobras f1_macro']}) · acumulado {reg['Δ acuracia (acum. vs 45 min)']:+6.2f}")
    incrementos = pd.DataFrame(incrementos)

    # ---- figura da progressão
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    x = tabela["Minutos observados"]
    ax.plot(x, tabela["acuracia"], "o-", color="#2C6350", label="Acurácia")
    ax.plot(x, tabela["f1_macro"], "s--", color="#9C6317", label="F1 macro")
    ax.axvline(45, color="#93372A", lw=1, ls=":")
    ax.annotate("intervalo\n(caso de uso)", xy=(45, ax.get_ylim()[0]), xytext=(46.5, ax.get_ylim()[0] + 1),
                fontsize=8, color="#93372A")
    for _, r in tabela.iterrows():
        ax.annotate(f"{r['acuracia']:.1f}", (r["Minutos observados"], r["acuracia"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(list(MINUTOS_OBSERVADOS.values()))
    ax.set_xlabel("Minutos de partida observados pelo modelo")
    ax.set_ylabel("Desempenho (%, validação cruzada)")
    ax.set_title("Progressão do desempenho conforme o modelo observa mais da partida")
    ax.legend(frameon=False)
    fig.tight_layout()
    destino = caminho(cfg, "figuras_dir")
    destino.mkdir(parents=True, exist_ok=True)
    for p in (destino / "wp5_progressao.png", run.arquivo("wp5_progressao.png")):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- tabelas e manifest
    escrever_latex(
        tabela.rename(columns={"acuracia": "Acurácia", "f1_macro": "F1 macro",
                               "f1_Derrota": "F1 Derr.", "f1_Empate": "F1 Emp.",
                               "f1_Vitória": "F1 Vit.", "log_loss": "Log-Loss",
                               "acuracia_default": "Acur. (default)",
                               "f1_macro_default": "F1 macro (default)"}),
        caminho(cfg, "tabelas_dir") / "wp5_progressao.tex",
        legenda=("Desempenho por quantidade de partida observada, sob validação cruzada de cinco "
                 "dobras agrupadas por partida em treino $\\cup$ validação. O conjunto de teste não "
                 "participa. Valores em \\%, exceto Log-Loss."),
        rotulo="tab:wp5-progressao",
    )
    escrever_latex(
        incrementos, caminho(cfg, "tabelas_dir") / "wp5_incrementos.tex",
        legenda=("Ganho por bloco adicional de 15 minutos, emparelhado nas mesmas dobras. "
                 "\\emph{bloco} é o ganho sobre a variante imediatamente anterior; "
                 "\\emph{acum.} é o ganho sobre o modelo que observa apenas o primeiro tempo."),
        rotulo="tab:wp5-incrementos",
    )
    tabela.to_csv(run.arquivo("progressao.csv"), index=False)
    incrementos.to_csv(run.arquivo("incrementos.csv"), index=False)
    run.escrever_manifest(
        wp5={
            "base": base.name,
            "diagnostico_2t": diag,
            "n_trials": n_trials,
            "metrica": metrica,
            "janela": cfg["otimizacao"]["janela"],
            "progressao": tabela.to_dict(orient="records"),
            "incrementos": incrementos.to_dict(orient="records"),
        }
    )
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n" + tabela.to_string(index=False))
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
