"""Os quatro pontos de comparação obrigatórios (WP1).

Por que quatro e não um:

* **acaso uniforme (33,3%)** — o acaso de verdade em três classes;
* **classe majoritária (~37,7%)** — o número que costuma ser citado como "acaso", mas que
  aqui é *artefato da construção espelhada da base*: cada partida gera duas linhas com
  alvos complementares, logo P(vitória) = P(derrota) exatamente. Precisa ser declarado
  como artefato no texto (achado A7);
* **mando de campo (~44,7%)** — a regra trivial "casa vence, fora perde". É o baseline mais
  duro e usa `is_home`, atributo que o modelo do TCC I descartava (achado A6);
* **modelo** — comparado contra os três acima, em todas as partições.

Uso:
    python -m src.modeling.baselines
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import caminho, carregar_config
from ..common.run import Run
from .splits import PARTICOES_INTERNAS, atribuir_particoes, carregar_dataset


def metricas(y_true: pd.Series, y_pred: np.ndarray, classes: dict[int, str]) -> dict[str, float]:
    """Acurácia e F1 por classe + macro. Implementado sobre a matriz de confusão para que
    baselines analíticos e empíricos usem exatamente a mesma definição."""
    rotulos = sorted(classes)
    conf = np.zeros((len(rotulos), len(rotulos)))
    for i, real in enumerate(rotulos):
        for j, prev in enumerate(rotulos):
            conf[i, j] = np.sum((y_true.to_numpy() == real) & (y_pred == prev))
    return _do_confusao(conf, classes)


def _do_confusao(conf: np.ndarray, classes: dict[int, str]) -> dict[str, float]:
    rotulos = sorted(classes)
    total = conf.sum()
    f1s = []
    saida: dict[str, float] = {"acuracia": round(100 * np.trace(conf) / total, 2)}
    for i, rotulo in enumerate(rotulos):
        vp, previstos, reais = conf[i, i], conf[:, i].sum(), conf[i, :].sum()
        prec = vp / previstos if previstos else 0.0
        rec = vp / reais if reais else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        saida[f"f1_{classes[rotulo]}"] = round(100 * f1, 2)
    saida["f1_macro"] = round(100 * float(np.mean(f1s)), 2)
    return saida


def baseline_acaso_uniforme(y_true: pd.Series, classes: dict[int, str]) -> dict[str, float]:
    """Analítico, via matriz de confusão esperada: C[i][j] = n_i / 3.

    Preferido a sortear rótulos: com ~340 partidas no teste, uma amostragem aleatória
    oscila ±2 p.p. e introduziria ruído num número que é exato por definição.
    """
    rotulos = sorted(classes)
    n_por_classe = np.array([(y_true == r).sum() for r in rotulos], dtype=float)
    conf = np.outer(n_por_classe, np.ones(len(rotulos))) / len(rotulos)
    return _do_confusao(conf, classes)


def baseline_classe_majoritaria(
    y_treino: pd.Series, y_true: pd.Series, classes: dict[int, str]
) -> dict[str, float]:
    """A classe é escolhida no TREINO — um baseline também não pode olhar o teste."""
    majoritaria = int(y_treino.value_counts().idxmax())
    return metricas(y_true, np.full(len(y_true), majoritaria), classes)


def baseline_mando_de_campo(sub: pd.DataFrame, alvo: str, classes: dict[int, str]) -> dict[str, float]:
    """`is_home` = 1 → prevê vitória (2); `is_home` = 0 → prevê derrota (0). Nunca prevê empate."""
    pred = np.where(sub["is_home"].to_numpy() == 1, 2, 0)
    return metricas(sub[alvo], pred, classes)


def tabela_baselines(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ds = cfg["dataset"]
    alvo = ds["alvo"]
    classes = {int(k): v for k, v in ds["classes"].items()}
    y_treino = df.loc[df["particao"] == "treino", alvo]

    ordem = list(PARTICOES_INTERNAS) + list(cfg["particao"]["validacao_externa"])
    linhas = []
    for nome in ordem:
        sub = df[df["particao"] == nome]
        if sub.empty:
            continue
        calculos = {
            "acaso uniforme": lambda: baseline_acaso_uniforme(sub[alvo], classes),
            "classe majoritária": lambda: baseline_classe_majoritaria(y_treino, sub[alvo], classes),
            "mando de campo": lambda: baseline_mando_de_campo(sub, alvo, classes),
        }
        for rotulo, fn in calculos.items():
            linhas.append(
                {
                    "Partição": nome,
                    "Baseline": rotulo,
                    "Partidas": sub[ds["chave_partida"]].nunique(),
                    **fn(),
                }
            )
    return pd.DataFrame(linhas)


def main() -> None:
    cfg = carregar_config()
    run = Run(cfg, nome="wp1_baselines")

    df = atribuir_particoes(carregar_dataset(cfg), cfg)
    tabela = tabela_baselines(df, cfg)

    from ..reporting.tables import escrever_latex

    escrever_latex(
        tabela,
        caminho(cfg, "tabelas_dir") / "baselines.tex",
        legenda=(
            "Baselines por partição (valores em \\%). O baseline de classe majoritária "
            "coincide com P(vitória) = P(derrota) por construção da base espelhada; o de "
            "mando de campo é a regra trivial \\emph{casa vence, fora perde}."
        ),
        rotulo="tab:baselines",
    )
    tabela.to_csv(run.arquivo("baselines.csv"), index=False)
    run.escrever_manifest(baselines=tabela.to_dict(orient="records"))

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(tabela.to_string(index=False))


if __name__ == "__main__":
    main()
