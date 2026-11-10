"""Guardas anti-vazamento da partição (WP0/WP1).

Cada teste aqui corresponde a um erro concreto que a auditoria de 10/08/2026 encontrou ou
que a estrutura da base permite cometer silenciosamente.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.common.config import carregar_config
from src.modeling.splits import (
    PARTICOES_INTERNAS,
    SEM_PARTICAO,
    atribuir_particoes,
    carregar_dataset,
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture(scope="module")
def df(cfg: dict) -> pd.DataFrame:
    return atribuir_particoes(carregar_dataset(cfg), cfg)


def test_toda_partida_recebe_exatamente_uma_particao(df: pd.DataFrame, cfg: dict) -> None:
    chave = cfg["dataset"]["chave_partida"]
    assert (df["particao"] != SEM_PARTICAO).all()
    assert df["particao"].isna().sum() == 0
    assert (df.groupby(chave)["particao"].nunique() == 1).all()


def test_nenhuma_partida_atravessa_particoes(df: pd.DataFrame, cfg: dict) -> None:
    chave = cfg["dataset"]["chave_partida"]
    conjuntos = {p: set(sub[chave]) for p, sub in df.groupby("particao")}
    nomes = sorted(conjuntos)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1 :]:
            comum = conjuntos[a] & conjuntos[b]
            assert not comum, f"{len(comum)} partidas em comum entre {a} e {b}"


def test_as_duas_linhas_de_uma_partida_ficam_juntas(df: pd.DataFrame, cfg: dict) -> None:
    """A7: se as linhas espelhadas caírem em partições diferentes, o alvo do teste é
    dedutível do treino através de `field_tilt = 1 - x`."""
    chave = cfg["dataset"]["chave_partida"]
    por_partida = df.groupby(chave).agg(n=("particao", "size"), particoes=("particao", "nunique"))
    assert (por_partida["n"] == cfg["dataset"]["linhas_por_partida"]).all()
    assert (por_partida["particoes"] == 1).all()


def test_ordem_temporal_estrita_entre_treino_validacao_teste(df: pd.DataFrame, cfg: dict) -> None:
    coluna_data = cfg["dataset"]["coluna_data"]
    interno = df[df["particao"].isin(PARTICOES_INTERNAS)]
    lim = interno.groupby("particao")[coluna_data].agg(["min", "max"])
    assert lim.loc["treino", "max"] < lim.loc["validacao", "min"]
    assert lim.loc["validacao", "max"] < lim.loc["teste", "min"]


def test_dominio_primario_e_homogeneo(df: pd.DataFrame, cfg: dict) -> None:
    """Treino, validação e teste têm de vir da MESMA população — é o ponto do achado A9."""
    prim = cfg["particao"]["dominio_primario"]
    interno = df[df["particao"].isin(PARTICOES_INTERNAS)]
    assert set(interno["competition_id"].unique()) == set(prim["competition_ids"])
    assert set(interno["season_id"].unique()) == set(prim["season_ids"])
    for particao in PARTICOES_INTERNAS:
        sub = interno[interno["particao"] == particao]
        assert set(sub["competition_id"].unique()) == set(prim["competition_ids"]), (
            f"{particao} não cobre as cinco ligas"
        )


def test_validacao_externa_nao_toca_o_dominio_primario(df: pd.DataFrame, cfg: dict) -> None:
    prim = cfg["particao"]["dominio_primario"]
    externos = df[df["particao"].isin(cfg["particao"]["validacao_externa"])]
    dentro = externos[
        externos["competition_id"].isin(prim["competition_ids"])
        & externos["season_id"].isin(prim["season_ids"])
    ]
    assert dentro.empty, f"{len(dentro)} linhas do domínio primário vazaram para validação externa"


def test_contagens_batem_com_o_config(df: pd.DataFrame, cfg: dict) -> None:
    """Se este teste falhar, algum número da monografia ficou obsoleto."""
    chave = cfg["dataset"]["chave_partida"]
    part = cfg["particao"]
    interno = df[df["particao"].isin(PARTICOES_INTERNAS)][chave].nunique()
    assert interno == part["dominio_primario"]["partidas_esperadas"]
    for nome, regra in part["validacao_externa"].items():
        obtido = df[df["particao"] == nome][chave].nunique()
        assert obtido == regra["partidas_esperadas"], f"{nome}: {obtido} != {regra['partidas_esperadas']}"


def test_particao_e_deterministica(cfg: dict) -> None:
    a = atribuir_particoes(carregar_dataset(cfg), cfg)
    b = atribuir_particoes(carregar_dataset(cfg), cfg)
    pd.testing.assert_series_equal(a["particao"], b["particao"])


def test_teste_e_a_ultima_janela_temporal_do_dominio_primario(df: pd.DataFrame, cfg: dict) -> None:
    """O teste tem de ser o futuro do domínio primário, não uma amostra aleatória dele."""
    coluna_data = cfg["dataset"]["coluna_data"]
    interno = df[df["particao"].isin(PARTICOES_INTERNAS)]
    assert (
        interno[interno["particao"] == "teste"][coluna_data].min()
        > interno[interno["particao"] != "teste"][coluna_data].max()
    )
