"""Guardas do WP3 (avaliação final).

Existem por causa de uma falha concreta: na primeira tentativa de abrir o teste, em 03/09/2026,
a regra de mando de campo estourou com `KeyError` porque guardava `is_home` indexado pelas linhas
da janela de treino, e as linhas do teste têm outros índices. O run morreu no meio, antes de
gravar ou exibir qualquer número.

O custo dessa classe de bug é desproporcional: o conjunto de teste é avaliado **uma única vez**,
e um erro de encanamento descoberto durante essa execução obriga a tocá-lo de novo. Estes testes
verificam que todo estimador declarado consegue prever em toda partição **fora da janela de
ajuste**, sem calcular nenhuma métrica — de modo que a suíte possa rodar quantas vezes for
preciso sem observar desempenho no teste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import carregar_config
from src.modeling.evaluate import localizar_run_wp2, montar_estimadores
from src.modeling.splits import atribuir_particoes, carregar_dataset
from src.modeling.train import janela_otimizacao, matrizes


@pytest.fixture(scope="module")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture(scope="module")
def df(cfg: dict) -> pd.DataFrame:
    return atribuir_particoes(carregar_dataset(cfg), cfg)


@pytest.fixture(scope="module")
def declarados(cfg: dict, df: pd.DataFrame) -> list[dict]:
    run_wp2 = localizar_run_wp2(cfg, None)
    decl, _ = montar_estimadores(cfg, janela_otimizacao(df, cfg), df, run_wp2)
    return decl


def particoes_fora_da_janela(cfg: dict) -> list[str]:
    return ["teste"] + list(cfg["particao"]["validacao_externa"])


def test_todos_os_estimadores_preveem_fora_da_janela(
    cfg: dict, df: pd.DataFrame, declarados: list[dict]
) -> None:
    """Nenhum estimador pode depender do índice das linhas de treino.

    Só formato e sanidade das probabilidades — nenhuma métrica de desempenho é calculada.
    """
    for nome in particoes_fora_da_janela(cfg):
        sub = df[df["particao"] == nome]
        assert not sub.empty, f"partição {nome} vazia"
        for d in declarados:
            X, _, _ = matrizes(sub, cfg, d["conjunto"])
            proba = d["prever"](X)
            assert proba.shape == (len(X), 3), f"{nome} / {d['Modelo']}: shape {proba.shape}"
            assert np.allclose(proba.sum(axis=1), 1.0), f"{nome} / {d['Modelo']}: não soma 1"
            assert np.isfinite(proba).all(), f"{nome} / {d['Modelo']}: valores não finitos"


def test_conjunto_de_modelos_declarado_esta_completo(cfg: dict, declarados: list[dict]) -> None:
    """O que é avaliado no teste tem de ser o que foi declarado antes de abri-lo."""
    rotulos = {(d["Modelo"], d["Atributos"], d["Otimizado para"]) for d in declarados}
    n_conjuntos = len(cfg["otimizacao"]["conjuntos"])
    n_metricas = len(cfg["otimizacao"]["metricas"])
    otimizados = {r for r in rotulos if r[0] == "XGBoost otimizado"}
    assert len(otimizados) == n_conjuntos * n_metricas
    for rotulo in ("XGBoost default", "regressão logística"):
        assert sum(1 for r in rotulos if r[0] == rotulo) == n_conjuntos
    for rotulo in ("classe majoritária", "mando de campo"):
        assert sum(1 for r in rotulos if r[0] == rotulo) == 1, f"{rotulo} deve aparecer uma vez"


def test_existe_exatamente_um_modelo_principal(declarados: list[dict]) -> None:
    principais = [d for d in declarados if d["principal"]]
    assert len(principais) == 1
    assert principais[0]["Modelo"] == "XGBoost otimizado"
    assert principais[0]["Otimizado para"] == "f1_macro"
