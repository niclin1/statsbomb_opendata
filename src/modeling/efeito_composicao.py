"""Quanto do desempenho reportado vem da composição do conjunto de teste.

Experimento controlado: **um único modelo, duas partições de teste**. Como o modelo é o mesmo,
os hiperparâmetros são os mesmos e o treino é o mesmo, toda diferença observada entre as duas
avaliações é atribuível à **composição do conjunto**, e não à qualidade do modelo.

As duas partições:

* **Teste correto (WP1)** — 340 partidas das últimas rodadas de 2015/16, mesmo domínio do treino
  (cinco grandes ligas europeias, temporada completa).
* **Teste do TCC I** — reconstruído exatamente como `src/train_optuna.py` o produzia: corte 80/20
  por índice de linha sobre a base ordenada por data, o que resulta em 527 partidas de
  2018-06-28 a 2024-07-15, formadas por torneios de seleções, Indian Super League e recortes de
  Barcelona, PSG e Leverkusen.

Por que isto **não** reabre o conjunto de teste reservado: as 527 partidas do teste antigo são
posteriores a 2018 e não têm interseção com as 340 do teste correto, nem com a janela de ajuste
(2015/16). Todas elas pertencem aos conjuntos de validação externa, já avaliados no WP3. Nenhuma
informação nova é consumida, e nenhum parâmetro do modelo é alterado.

O resultado quantifica um viés que raramente é medido: **quanto de um desempenho publicado decorre
de onde ele foi medido.**

Uso:
    python -m src.modeling.efeito_composicao
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from ..common.config import caminho, carregar_config, features
from ..common.run import Run
from ..reporting.tables import escrever_latex
from .baselines import metricas
from .evaluate import ic_bootstrap, localizar_run_wp2
from .splits import atribuir_particoes, carregar_dataset
from .train import matrizes

FRACAO_TREINO_TCC1 = 0.8


def particao_tcc1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reconstrói o conjunto de teste do TCC I: corte 80/20 por índice de linha.

    Reproduz `src/train_optuna.py`: ordenação por `match_date` e
    `split_idx = int(len(X) * 0.8)`, sem agrupamento por partida e sem corte por data.
    """
    ds = cfg["dataset"]
    ordenado = df.sort_values(ds["coluna_data"]).reset_index(drop=True)
    corte = int(len(ordenado) * FRACAO_TREINO_TCC1)
    return ordenado.iloc[corte:].copy()


def avaliar(bst: xgb.Booster, sub: pd.DataFrame, cols: list[str], cfg: dict,
            classes: dict[int, str]) -> dict:
    X, y, grupos = matrizes(sub, cfg, cfg["features"]["conjunto_principal"])
    proba = bst.predict(xgb.DMatrix(X[cols]))
    m = metricas(y, np.argmax(proba, axis=1), classes)
    ic = ic_bootstrap(y, proba, grupos, classes, cfg["seed"])
    return {**m, "ic_acuracia": ic["acuracia"], "n_partidas": int(sub[cfg["dataset"]["chave_partida"]].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser(description="efeito da composição do conjunto de teste")
    ap.add_argument("--run-wp2", default=None)
    args = ap.parse_args()

    cfg = carregar_config()
    classes = {int(k): v for k, v in cfg["dataset"]["classes"].items()}
    conjunto = cfg["features"]["conjunto_principal"]
    cols = features(cfg, conjunto)

    run = Run(cfg, nome="efeito_composicao")
    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    run_wp2 = localizar_run_wp2(cfg, args.run_wp2)
    bst = xgb.Booster()
    bst.load_model(run_wp2 / f"modelo_{conjunto}_{cfg['otimizacao']['metrica_principal']}.ubj")

    antigo = particao_tcc1(df, cfg)
    correto = df[df["particao"] == "teste"]
    ds = cfg["dataset"]

    print("== reconstrução do conjunto de teste do TCC I ==")
    print(f"  partidas: {antigo[ds['chave_partida']].nunique()} | linhas: {len(antigo)}")
    print(f"  período: {antigo[ds['coluna_data']].min().date()} → {antigo[ds['coluna_data']].max().date()}")
    comp = antigo.groupby("competicao")[ds["chave_partida"]].nunique().sort_values(ascending=False)
    print(f"  composição: {comp.to_dict()}")
    sobreposicao = set(antigo[ds["chave_partida"]]) & set(correto[ds["chave_partida"]])
    print(f"  interseção com o teste correto: {len(sobreposicao)} partidas")
    assert not sobreposicao, "as duas partições não podem se sobrepor"

    print("\n== mesmo modelo, duas partições ==")
    linhas = []
    for rotulo, sub in [("Teste correto (WP1)", correto), ("Teste da etapa anterior (reconstruído)", antigo)]:
        r = avaliar(bst, sub, cols, cfg, classes)
        linhas.append({
            "Conjunto de teste": rotulo,
            "Partidas": r["n_partidas"],
            "Acurácia": r["acuracia"],
            "IC 95%": f"[{r['ic_acuracia'][0]}; {r['ic_acuracia'][1]}]",
            "F1 macro": r["f1_macro"],
            **{f"F1 {c}": r[f"f1_{c}"] for c in classes.values()},
        })
        print(f"  {rotulo:32s} n={r['n_partidas']:3d}  acurácia {r['acuracia']:5.2f} "
              f"[{r['ic_acuracia'][0]}; {r['ic_acuracia'][1]}]  F1 macro {r['f1_macro']:5.2f}")

    dif = linhas[1]["Acurácia"] - linhas[0]["Acurácia"]
    print(f"\n  DIFERENÇA ATRIBUÍVEL À COMPOSIÇÃO: {dif:+.2f} p.p.")
    print("  (mesmo modelo, mesmos hiperparâmetros, mesmo treino)")

    # decomposição do teste antigo por bloco de origem
    print("\n== de onde vem a inflação: teste do TCC I por competição ==")
    decomp = []
    for comp_nome, grupo in antigo.groupby("competicao"):
        if grupo[ds["chave_partida"]].nunique() < 20:
            continue
        r = avaliar(bst, grupo, cols, cfg, classes)
        maj = grupo[ds["alvo"]].value_counts(normalize=True).max() * 100
        decomp.append({"Competição": comp_nome, "Partidas": r["n_partidas"],
                       "Acurácia": r["acuracia"], "Classe majoritária": round(float(maj), 2),
                       "Margem sobre a majoritária": round(r["acuracia"] - float(maj), 2)})
        print(f"  {comp_nome:22s} n={r['n_partidas']:3d}  acurácia {r['acuracia']:5.2f}  "
              f"classe majoritária {maj:5.2f}  margem {r['acuracia']-maj:+5.2f}")
    decomp = pd.DataFrame(decomp).sort_values("Acurácia", ascending=False)

    # Legendas e rótulo de linha usam "etapa anterior deste trabalho", a forma do texto da
    # monografia. As tabelas em outputs/tabelas/ foram escritas antes da troca e ainda dizem
    # "TCC I"; a etapa tipográfica (src/reporting/formatar_tabelas_tex.py) faz a troca no build.
    tabela = pd.DataFrame(linhas)
    escrever_latex(
        tabela, caminho(cfg, "tabelas_dir") / "efeito_composicao.tex",
        legenda=("Desempenho do mesmo modelo final em duas partições de teste. A partição usada na "
                 "etapa anterior deste trabalho foi reconstruída pelo corte 80/20 por índice de "
                 "linha; a partição correta segue o "
                 "protocolo do Capítulo de metodologia. Como o modelo, os hiperparâmetros e o "
                 "treino são idênticos, a diferença é atribuível à composição do conjunto."),
        rotulo="tab:efeito-composicao",
    )
    escrever_latex(
        decomp, caminho(cfg, "tabelas_dir") / "efeito_composicao_decomposicao.tex",
        legenda=("Decomposição por competição do conjunto de teste da etapa anterior deste trabalho. "
                 "A margem sobre a classe majoritária indica quanto o modelo acrescenta em cada bloco."),
        rotulo="tab:efeito-composicao-decomp",
    )
    tabela.to_csv(run.arquivo("comparativo.csv"), index=False)
    decomp.to_csv(run.arquivo("decomposicao.csv"), index=False)
    run.escrever_manifest(
        efeito_composicao={
            "modelo": f"modelo_{conjunto}_{cfg['otimizacao']['metrica_principal']}.ubj",
            "run_wp2": run_wp2.name,
            "particao_tcc1": {
                "partidas": int(antigo[ds["chave_partida"]].nunique()),
                "periodo": [str(antigo[ds["coluna_data"]].min().date()),
                            str(antigo[ds["coluna_data"]].max().date())],
                "composicao": {k: int(v) for k, v in comp.items()},
            },
            "comparativo": tabela.to_dict(orient="records"),
            "diferenca_atribuivel_a_composicao_pp": round(float(dif), 2),
            "decomposicao": decomp.to_dict(orient="records"),
        }
    )
    print(f"\nrun: {run.diretorio}")


if __name__ == "__main__":
    main()
