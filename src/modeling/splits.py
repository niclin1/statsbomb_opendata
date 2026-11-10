"""Partição treino / validação / teste — ÚNICA fonte de verdade (WP1).

Protocolo decidido em 10/08/2026 (ver _planning/auditoria_dados_2026-08-10.md):

* **Domínio primário**: cinco grandes ligas europeias, temporada 2015/16 — o único bloco
  da base que é uma população homogênea e completa (as temporadas de clubes pós-2016 são
  recortes de uma equipe só: Barcelona, PSG, Leverkusen — achado A9).
* **Split temporal por data** dentro da temporada: o teste são as últimas rodadas, com o
  modelo treinado nas anteriores. Corresponde ao caso de uso (prever no intervalo).
  O corte é por data, então nenhuma rodada é dividida entre partições.
* **Agrupamento por `match_id` é obrigatório** (achado A7): cada partida gera duas linhas
  espelhadas, com `field_tilt_A + field_tilt_B = 1,0` e alvos complementares. Separar as
  duas linhas de uma partida vaza o alvo do teste através de `1 - x`.
* **Quatro conjuntos de validação externa**, nunca usados em treino nem na seleção de
  hiperparâmetros; cada um mede um deslocamento de distribuição diferente.

Uso:
    python -m src.modeling.splits          # gera tabela de composição + manifest do run
"""
from __future__ import annotations

import pandas as pd

from ..common.config import (
    caminho,
    carregar_config,
    dominio_competicao,
    features,
    nome_competicao,
)
from ..common.run import Run

PARTICOES_INTERNAS = ("treino", "validacao", "teste")
SEM_PARTICAO = "nao_atribuido"


# --------------------------------------------------------------------------- dados


def carregar_dataset(cfg: dict, arquivo=None) -> pd.DataFrame:
    """Lê a base e acrescenta colunas derivadas de leitura (nome e domínio da competição).

    `arquivo` permite apontar para uma base alternativa — usado pelo WP5, que roda sobre
    `data/processed/momentum_v5_*.parquet` (mesmas partidas, com os blocos do 2º tempo).
    Como a partição é derivada de `competition_id`, `season_id` e da data, ela sai idêntica,
    e os testes de contagem do WP0 continuam valendo como verificação.
    """
    ds = cfg["dataset"]
    origem = arquivo if arquivo is not None else caminho(cfg, "dataset")
    if str(origem).endswith(".parquet"):
        df = pd.read_parquet(origem)
        df[ds["coluna_data"]] = pd.to_datetime(df[ds["coluna_data"]])
    else:
        df = pd.read_csv(origem, parse_dates=[ds["coluna_data"]])
    df["competicao"] = df["competition_id"].map(lambda c: nome_competicao(cfg, c))
    df["dominio"] = df["competition_id"].map(lambda c: dominio_competicao(cfg, c))
    return df.sort_values([ds["coluna_data"], ds["chave_partida"], "is_home"]).reset_index(drop=True)


def _por_partida(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Uma linha por partida — a unidade de partição (A7: n efetivo = nº de partidas)."""
    ds = cfg["dataset"]
    return (
        df.drop_duplicates(ds["chave_partida"])[
            [ds["chave_partida"], ds["coluna_data"], "competition_id", "season_id"]
        ]
        .sort_values(ds["coluna_data"])
        .reset_index(drop=True)
    )


def _mascara_regra(partidas: pd.DataFrame, regra: dict, coluna_data: str) -> pd.Series:
    """Aplica uma regra declarativa do config (competition_ids / season_ids / data_min)."""
    m = partidas["competition_id"].isin(regra["competition_ids"])
    if "season_ids" in regra:
        m &= partidas["season_id"].isin(regra["season_ids"])
    if "data_min" in regra:
        m &= partidas[coluna_data] >= pd.Timestamp(regra["data_min"])
    if "data_max" in regra:
        m &= partidas[coluna_data] <= pd.Timestamp(regra["data_max"])
    return m


# ----------------------------------------------------------------------- partição


def cortes_temporais(datas: pd.Series, fracoes: dict[str, float]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Datas de corte treino|validação e validação|teste.

    O corte é feito sobre a distribuição acumulada de partidas por *data*, e não por
    índice de linha: todas as partidas de um mesmo dia caem na mesma partição. Como
    consequência, as frações realizadas não batem exatamente com as pedidas — é o preço
    de não dividir uma rodada, e as frações reais são reportadas na tabela de composição.
    """
    por_dia = datas.value_counts().sort_index()
    acumulado = por_dia.cumsum() / por_dia.sum()
    d1 = acumulado.index[acumulado >= fracoes["treino"]][0]
    d2 = acumulado.index[acumulado >= fracoes["treino"] + fracoes["validacao"]][0]
    return d1, d2


def atribuir_particoes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Devolve `df` com a coluna `particao`. Determinístico e sem estado."""
    ds, part_cfg = cfg["dataset"], cfg["particao"]
    chave, coluna_data = ds["chave_partida"], ds["coluna_data"]

    partidas = _por_partida(df, cfg)
    particao = pd.Series(SEM_PARTICAO, index=partidas.index, dtype="object")

    # 1) domínio primário: split temporal por data
    prim = part_cfg["dominio_primario"]
    m_prim = _mascara_regra(partidas, prim, coluna_data)
    d1, d2 = cortes_temporais(partidas.loc[m_prim, coluna_data], prim["split"]["fracoes"])
    datas = partidas[coluna_data]
    particao[m_prim & (datas <= d1)] = "treino"
    particao[m_prim & (datas > d1) & (datas <= d2)] = "validacao"
    particao[m_prim & (datas > d2)] = "teste"

    # 2) conjuntos de validação externa
    for nome, regra in part_cfg["validacao_externa"].items():
        m_ext = _mascara_regra(partidas, regra, coluna_data) & (particao == SEM_PARTICAO)
        particao[m_ext] = nome

    partidas = partidas.assign(particao=particao)
    _validar(partidas, cfg, d1, d2)

    return df.merge(partidas[[chave, "particao"]], on=chave, how="left", validate="many_to_one")


def _validar(partidas: pd.DataFrame, cfg: dict, d1: pd.Timestamp, d2: pd.Timestamp) -> None:
    """Falha alto e cedo. Estas invariantes também são testadas em tests/test_splits.py."""
    part_cfg = cfg["particao"]
    coluna_data = cfg["dataset"]["coluna_data"]

    orfas = partidas[partidas["particao"] == SEM_PARTICAO]
    if not orfas.empty:
        pares = orfas.groupby(["competition_id", "season_id"]).size().to_dict()
        raise ValueError(
            f"{len(orfas)} partidas não pertencem a nenhuma partição: {pares}. "
            "Acrescente uma regra em config/experiment.yaml — nenhuma partida pode ficar de fora "
            "sem decisão explícita."
        )

    if not (d1 < d2):
        raise ValueError(f"cortes temporais inválidos: {d1} !< {d2}")

    esperados = {"__primario__": part_cfg["dominio_primario"]["partidas_esperadas"]}
    esperados.update(
        {n: r["partidas_esperadas"] for n, r in part_cfg["validacao_externa"].items()}
    )
    contagem = partidas["particao"].value_counts().to_dict()
    obtidos = {"__primario__": sum(contagem.get(p, 0) for p in PARTICOES_INTERNAS)}
    obtidos.update({n: contagem.get(n, 0) for n in part_cfg["validacao_externa"]})
    divergencias = {k: (esperados[k], obtidos[k]) for k in esperados if esperados[k] != obtidos[k]}
    if divergencias:
        raise ValueError(
            "nº de partidas divergente do declarado em config (esperado, obtido): "
            f"{divergencias}. Se a base mudou (reexecução do ETL), atualize o config "
            "e o texto da monografia no mesmo commit."
        )

    interno = partidas[partidas["particao"].isin(PARTICOES_INTERNAS)]
    limites = interno.groupby("particao")[coluna_data].agg(["min", "max"])
    if not (
        limites.loc["treino", "max"] < limites.loc["validacao", "min"]
        and limites.loc["validacao", "max"] < limites.loc["teste", "min"]
    ):
        raise ValueError(f"ordem temporal violada entre partições:\n{limites}")


# ------------------------------------------------------------------------ relatos


def resumo_particoes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tabela de composição das partições — entrega do WP1."""
    ds = cfg["dataset"]
    chave, coluna_data, alvo = ds["chave_partida"], ds["coluna_data"], ds["alvo"]
    classes = {int(k): v for k, v in ds["classes"].items()}
    total_partidas = df[chave].nunique()

    ordem = list(PARTICOES_INTERNAS) + list(cfg["particao"]["validacao_externa"])
    linhas = []
    for nome in ordem:
        sub = df[df["particao"] == nome]
        dist = sub[alvo].value_counts(normalize=True).mul(100)
        comps = sub.groupby("competicao")[chave].nunique().sort_values(ascending=False)
        linhas.append(
            {
                "Partição": nome,
                "Partidas": sub[chave].nunique(),
                "Linhas": len(sub),
                "% da base": round(100 * sub[chave].nunique() / total_partidas, 1),
                "Início": sub[coluna_data].min().date(),
                "Fim": sub[coluna_data].max().date(),
                **{
                    f"% {classes[c]}": round(dist.get(c, 0.0), 1)
                    for c in sorted(classes)
                },
                "Competições": ", ".join(f"{c} ({n})" for c, n in comps.items()),
            }
        )
    return pd.DataFrame(linhas)


def main() -> None:
    cfg = carregar_config()
    run = Run(cfg, nome="wp1_particoes")

    df = carregar_dataset(cfg)
    df = atribuir_particoes(df, cfg)
    resumo = resumo_particoes(df, cfg)

    from ..reporting.tables import escrever_latex

    tabelas = caminho(cfg, "tabelas_dir")
    tabelas.mkdir(parents=True, exist_ok=True)
    colunas_tex = [c for c in resumo.columns if c != "Competições"]
    escrever_latex(
        resumo[colunas_tex],
        tabelas / "particoes.tex",
        legenda=(
            "Composição das partições. Domínio primário: cinco grandes ligas europeias, "
            "temporada 2015/16, com corte temporal por data (nenhuma rodada dividida). "
            "Os quatro conjuntos externos não participam do treino nem da seleção de "
            "hiperparâmetros."
        ),
        rotulo="tab:particoes",
    )
    resumo.to_csv(run.arquivo("particoes.csv"), index=False)

    prim = cfg["particao"]["dominio_primario"]
    d1, d2 = cortes_temporais(
        _por_partida(df, cfg)[cfg["dataset"]["coluna_data"]][
            _mascara_regra(_por_partida(df, cfg), prim, cfg["dataset"]["coluna_data"])
        ],
        prim["split"]["fracoes"],
    )
    run.escrever_manifest(
        particao={
            "corte_treino_validacao": str(d1.date()),
            "corte_validacao_teste": str(d2.date()),
            "fracoes_pedidas": prim["split"]["fracoes"],
            "fracoes_realizadas": {
                p: round(
                    df[df["particao"] == p][cfg["dataset"]["chave_partida"]].nunique()
                    / prim["partidas_esperadas"],
                    4,
                )
                for p in PARTICOES_INTERNAS
            },
            "resumo": resumo.drop(columns=["Competições"]).to_dict(orient="records"),
        },
        features={
            "conjunto_principal": cfg["features"]["conjunto_principal"],
            "n_atributos": len(features(cfg)),
            "nomes": features(cfg),
        },
    )

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(resumo.drop(columns=["Competições"]).to_string(index=False))
    print(f"\ncorte treino|validação: {d1.date()}   corte validação|teste: {d2.date()}")
    print(f"atributos ({len(features(cfg))}): {', '.join(features(cfg))}")
    print(f"\nrun: {run.diretorio.relative_to(caminho(cfg, 'runs_dir').parents[1])}")


if __name__ == "__main__":
    main()
