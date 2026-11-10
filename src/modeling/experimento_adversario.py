"""Diagnóstico: quanto custa o modelo não enxergar o xG e o PPDA do adversário.

Origem do achado (varredura de 03/09/2026). O texto do TCC I declara **18 atributos** e descreve
"cálculo isolado de xG, PPDA, Field Tilt e indicador de cartão vermelho em cada bloco". A
aritmética que fecha 18 é **3 métricas × 3 blocos × 2 lados** (equipe e adversário). O que foi
implementado é 3 × 3 × **um lado** + 2 contagens de cartão = 11.

Ou seja: o modelo observa quanto a própria equipe criou, e **não** quanto sofreu. O cartão
vermelho é a única métrica com contrapartida do adversário (`opp_red_cards_p1`), o que indica que
a intenção existia e não chegou aos atributos táticos.

Os atributos do adversário são deriváveis sem reprocessar o ETL: a base é espelhada (achado A7),
então o valor do adversário para uma partida é o valor da outra linha da mesma partida.

**Field Tilt do adversário é omitido de propósito.** Ele é redundante por construção: as duas
linhas somam exatamente 1,0 em 100% das partidas, logo `field_tilt_adv = 1 − field_tilt`. Incluí-lo
acrescentaria uma coluna colinear, não informação. Restam 6 atributos novos (xG e PPDA do
adversário em três blocos), totalizando 17.

**Este é um experimento diagnóstico, não uma mudança do modelo final.** Roda por validação cruzada
sobre `treino ∪ validação`; o conjunto de teste não é tocado. Serve para responder com número à
reconciliação "18 declarados × 11 implementados" e para dimensionar um item de trabalhos futuros.

Uso:
    python -m src.modeling.experimento_adversario
"""
from __future__ import annotations

import argparse

import optuna
import pandas as pd

from ..common.config import caminho, carregar_config, features
from ..common.run import Run
from ..reporting.tables import escrever_latex
from .splits import atribuir_particoes, carregar_dataset
from .train import avaliar_cv, dobras, estimador_xgb, janela_otimizacao, rodar_estudo

BLOCOS = ("0_15", "15_30", "30_45")
METRICAS_ADVERSARIO = ("xg", "ppda")  # field_tilt é redundante: soma 1 com o da equipe


def acrescentar_adversario(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Deriva xG e PPDA do adversário a partir da linha espelhada da mesma partida."""
    chave = cfg["dataset"]["chave_partida"]
    cols = [f"{m}_{b}" for m in METRICAS_ADVERSARIO for b in BLOCOS]

    espelho = df[[chave, "team_name", *cols]].rename(
        columns={"team_name": "_adversario", **{c: f"adv_{c}" for c in cols}}
    )
    juntado = df.merge(espelho, on=chave, how="inner")
    juntado = juntado[juntado["team_name"] != juntado["_adversario"]].drop(columns=["_adversario"])
    assert len(juntado) == len(df), "junção espelhada não preservou o número de linhas"
    return juntado.reset_index(drop=True), [f"adv_{c}" for c in cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="diagnóstico: atributos do adversário")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    cfg = carregar_config()
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    metrica = cfg["otimizacao"]["metrica_principal"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run = Run(cfg, nome="diag_atributos_adversario")
    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    df, cols_adv = acrescentar_adversario(df, cfg)
    sub = janela_otimizacao(df, cfg)

    base = features(cfg, "tatico_11")
    conjuntos = {"tatico_11 (atual)": base, "tatico_17 (+ adversário)": base + cols_adv}
    print(f"janela: {sub[cfg['dataset']['chave_partida']].nunique()} partidas · "
          f"{args.n_trials} trials por conjunto · métrica {metrica}")
    print(f"atributos do adversário derivados: {', '.join(cols_adv)}\n")

    linhas, por_dobra = [], {}
    for rotulo, cols in conjuntos.items():
        X, y = sub[cols], sub[cfg["dataset"]["alvo"]]
        grupos = sub[cfg["dataset"]["chave_partida"]].to_numpy()
        lista_dobras = dobras(X, y, grupos, cfg)
        estudo = rodar_estudo(
            cfg, X, y, lista_dobras, classes, args.n_trials,
            caminho_db=run.arquivo(f"study_{len(cols)}.db").as_posix(),
            nome=f"xgb_adv_{len(cols)}", metrica=metrica,
        )
        r = avaliar_cv(estimador_xgb(cfg, estudo.best_params), X, y, lista_dobras, classes)
        por_dobra[rotulo] = r["por_dobra"]
        linhas.append({"Conjunto": rotulo, "Atributos": len(cols),
                       **{k: r["media"][k] for k in ("acuracia", "f1_macro")},
                       **{f"f1_{c}": r["media"][f"f1_{c}"] for c in classes.values()},
                       "log_loss": r["media"]["log_loss"]})
        print(f"  {rotulo:26s} ({len(cols):2d} atrib.): acurácia {r['media']['acuracia']:.2f} · "
              f"F1 macro {r['media']['f1_macro']:.2f} · F1 empate {r['media'][f'f1_{classes[1]}']:.2f}")

    tabela = pd.DataFrame(linhas)
    a, b = (pd.DataFrame(por_dobra[k]) for k in conjuntos)
    print("\n== ganho emparelhado nas mesmas 5 dobras ==")
    ganhos = {}
    for m in ("acuracia", "f1_macro", "f1_Empate", "f1_Derrota", "f1_Vitória"):
        dif = b[m] - a[m]
        ganhos[m] = {"media": round(float(dif.mean()), 2),
                     "dobras_a_favor": f"{int((dif > 0).sum())}/5"}
        print(f"  {m:12s}: {a[m].mean():6.2f} → {b[m].mean():6.2f} | "
              f"dif {dif.mean():+6.2f} | dobras a favor: {int((dif > 0).sum())}/5")

    escrever_latex(
        tabela, caminho(cfg, "tabelas_dir") / "diag_atributos_adversario.tex",
        legenda=("Efeito de acrescentar xG e PPDA do adversário por bloco, sob validação cruzada de "
                 "cinco dobras agrupadas por partida em treino $\\cup$ validação. O conjunto de teste "
                 "não participa. O Field Tilt do adversário é omitido por ser redundante: as duas "
                 "equipes somam 1,0 por construção."),
        rotulo="tab:diag-adversario",
    )
    tabela.to_csv(run.arquivo("comparativo.csv"), index=False)
    run.escrever_manifest(
        diagnostico={"motivacao": "reconciliação 18 declarados × 11 implementados",
                     "colunas_derivadas": cols_adv, "n_trials": args.n_trials,
                     "comparativo": tabela.to_dict(orient="records"), "ganho_emparelhado": ganhos}
    )
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
