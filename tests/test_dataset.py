"""Invariantes da base (WP0). Rodam antes de qualquer treino.

Estes testes existem para travar a classe de erro que atravessou o TCC I: número no texto
divergindo do número nos dados. Se o ETL for reexecutado e a base mudar, eles falham *de
propósito* — o config e o texto precisam ser atualizados no mesmo commit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import carregar_config, features
from src.modeling.splits import carregar_dataset


@pytest.fixture(scope="module")
def cfg() -> dict:
    return carregar_config()


@pytest.fixture(scope="module")
def df(cfg: dict) -> pd.DataFrame:
    return carregar_dataset(cfg)


def test_tamanho_e_periodo_conferem_com_config(df: pd.DataFrame, cfg: dict) -> None:
    esperado, ds = cfg["dataset"]["esperado"], cfg["dataset"]
    assert len(df) == esperado["linhas"]
    assert df[ds["chave_partida"]].nunique() == esperado["partidas"]
    assert str(df[ds["coluna_data"]].min().date()) == esperado["data_min"]
    assert str(df[ds["coluna_data"]].max().date()) == esperado["data_max"]


def test_sem_nulos(df: pd.DataFrame) -> None:
    assert int(df.isna().sum().sum()) == 0


def test_duas_linhas_por_partida(df: pd.DataFrame, cfg: dict) -> None:
    ds = cfg["dataset"]
    contagem = df.groupby(ds["chave_partida"]).size()
    assert set(contagem.unique()) == {ds["linhas_por_partida"]}
    assert not df.duplicated([ds["chave_partida"], "team_name"]).any()


def test_estrutura_espelhada_field_tilt_soma_um(df: pd.DataFrame, cfg: dict) -> None:
    """A7: `field_tilt` de um lado é 1 menos o do outro. É por isso que a partição tem de
    agrupar por partida — separar as duas linhas vaza o alvo via `1 - x`."""
    chave = cfg["dataset"]["chave_partida"]
    for bloco in ("0_15", "15_30", "30_45"):
        soma = df.groupby(chave)[f"field_tilt_{bloco}"].sum()
        assert np.allclose(soma.to_numpy(), 1.0), f"field_tilt_{bloco} não soma 1 por partida"


def test_alvos_complementares(df: pd.DataFrame, cfg: dict) -> None:
    """A7: vitória+derrota ou empate+empate — a soma dos dois alvos é sempre 2, o que torna
    P(vitória) = P(derrota) um artefato de construção, não um fato observado."""
    ds = cfg["dataset"]
    soma = df.groupby(ds["chave_partida"])[ds["alvo"]].sum()
    assert set(soma.unique()) == {2}


def test_conjuntos_de_features_existem_e_tem_o_tamanho_declarado(df: pd.DataFrame, cfg: dict) -> None:
    """Trava o erro "18 atributos" do TCC I: os nomes vêm do config e têm de existir na base."""
    tamanhos = {"tatico_11": 11, "completo_12": 12}
    for conjunto, n in tamanhos.items():
        nomes = features(cfg, conjunto)
        assert len(nomes) == n, f"{conjunto} deveria ter {n} atributos, tem {len(nomes)}"
        assert len(set(nomes)) == len(nomes), f"{conjunto} tem atributo repetido"
        faltando = [c for c in nomes if c not in df.columns]
        assert not faltando, f"atributos ausentes na base: {faltando}"
    assert "is_home" in features(cfg, "completo_12")
    assert "is_home" not in features(cfg, "tatico_11")


def test_conjunto_principal_e_o_declarado_na_monografia(cfg: dict) -> None:
    """D2 revisada em 11/08/2026: o modelo principal usa 11 atributos táticos.

    `is_home` tem efeito nulo (−0,20 p.p. de F1 macro, 3/5 dobras) porque os atributos
    táticos já codificam o mando de campo (AUC 0,674). Este teste trava o número que vai
    para o texto — se alguém trocar `conjunto_principal` no config, a suíte avisa.
    """
    assert cfg["features"]["conjunto_principal"] == "tatico_11"
    assert len(features(cfg)) == 11


def test_alvo_tem_apenas_as_tres_classes_declaradas(df: pd.DataFrame, cfg: dict) -> None:
    ds = cfg["dataset"]
    assert set(df[ds["alvo"]].unique()) == {int(k) for k in ds["classes"]}
