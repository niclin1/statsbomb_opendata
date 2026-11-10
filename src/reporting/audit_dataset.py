"""Regenera os números do documento de auditoria (`_planning/auditoria_dados_2026-08-10.md`).

Existe para que o próprio diagnóstico seja reproduzível: qualquer número citado naquele
documento tem de sair daqui, não de uma inspeção manual perdida num chat.

Uso:
    python -m src.reporting.audit_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import carregar_config
from ..common.run import Run
from ..modeling.splits import carregar_dataset

BLOCOS = ("0_15", "15_30", "30_45")


def diagnostico(df: pd.DataFrame, cfg: dict) -> dict:
    ds = cfg["dataset"]
    chave, coluna_data, alvo = ds["chave_partida"], ds["coluna_data"], ds["alvo"]

    zero_xg = {b: round(100 * float((df[f"xg_{b}"] == 0).mean()), 2) for b in BLOCOS}
    tilt_zero = {b: round(100 * float((df[f"field_tilt_{b}"] == 0).mean()), 2) for b in BLOCOS}
    ppda_sentinela = {b: int((df[f"ppda_{b}"] == 50.0).sum()) for b in BLOCOS}

    clubes = df[df["dominio"] == "clubes"]
    selecoes = df[df["dominio"] == "selecoes"]

    def taxa_mando(sub: pd.DataFrame) -> dict:
        tab = pd.crosstab(sub["is_home"], sub[alvo], normalize="index").mul(100).round(2)
        return {int(k): {int(c): float(v) for c, v in linha.items()} for k, linha in tab.iterrows()}

    por_ano = (
        df.drop_duplicates(chave)
        .assign(ano=lambda d: d[coluna_data].dt.year)
        .pivot_table(index="ano", columns="dominio", values=chave, aggfunc="count", fill_value=0)
    )

    return {
        "tamanho": {
            "linhas": len(df),
            "partidas": int(df[chave].nunique()),
            "n_efetivo": int(df[chave].nunique()),  # A7: não são 5.270 observações independentes
            "data_min": str(df[coluna_data].min().date()),
            "data_max": str(df[coluna_data].max().date()),
            "nulos": int(df.isna().sum().sum()),
        },
        "A7_estrutura_espelhada": {
            "field_tilt_soma_1_por_partida_%": {
                b: round(
                    100
                    * float(np.isclose(df.groupby(chave)[f"field_tilt_{b}"].sum().to_numpy(), 1.0).mean()),
                    2,
                )
                for b in BLOCOS
            },
            "soma_dos_alvos_por_partida": df.groupby(chave)[alvo].sum().value_counts().to_dict(),
            "alvo_%_base_inteira": (df[alvo].value_counts(normalize=True).sort_index() * 100)
            .round(2)
            .to_dict(),
        },
        "A6_mando_de_campo": {
            "clubes_%_por_mando": taxa_mando(clubes),
            "selecoes_%_por_mando": taxa_mando(selecoes),
            # regra trivial: is_home == 1 -> prevê vitória (2); is_home == 0 -> prevê derrota (0)
            "acuracia_regra_trivial_clubes_%": round(
                100
                * float(
                    (
                        ((clubes["is_home"] == 1) & (clubes[alvo] == 2))
                        | ((clubes["is_home"] == 0) & (clubes[alvo] == 0))
                    ).mean()
                ),
                2,
            ),
        },
        "A9_desbalanceamento_temporal": {
            "partidas_por_ano_e_dominio": por_ano.to_dict(),
            "%_clubes_ate_2016_05": round(
                100
                * float(
                    (clubes.drop_duplicates(chave)[coluna_data] <= pd.Timestamp("2016-05-31")).mean()
                ),
                1,
            ),
            "composicao_clubes_pos_2016": {
                f"{comp}_s{sid}": {
                    "partidas": int(sub[chave].nunique()),
                    "equipe_dominante": sub["team_name"].value_counts().idxmax(),
                    "partidas_da_dominante": int(sub["team_name"].value_counts().max()),
                }
                for (comp, sid), sub in clubes[
                    clubes[coluna_data] >= pd.Timestamp("2016-06-01")
                ].groupby(["competicao", "season_id"])
            },
        },
        "sanidade_features": {
            "xg_zero_%": zero_xg,
            "field_tilt_zero_%": tilt_zero,
            "ppda_sentinela_50_linhas": ppda_sentinela,
            "ppda_mediana": {b: round(float(df[f"ppda_{b}"].median()), 2) for b in BLOCOS},
            "cartao_vermelho_p1_%_linhas": round(100 * float((df["team_red_cards_p1"] > 0).mean()), 2),
        },
    }


def main() -> None:
    cfg = carregar_config()
    run = Run(cfg, nome="wp0_auditoria")
    df = carregar_dataset(cfg)
    dados = diagnostico(df, cfg)
    run.escrever_json("auditoria.json", dados)
    run.escrever_manifest(auditoria=dados)

    t, a7, a6, a9, sf = (
        dados["tamanho"],
        dados["A7_estrutura_espelhada"],
        dados["A6_mando_de_campo"],
        dados["A9_desbalanceamento_temporal"],
        dados["sanidade_features"],
    )
    print(f"base: {t['linhas']} linhas / {t['partidas']} partidas "
          f"({t['data_min']} -> {t['data_max']}), {t['nulos']} nulos")
    print(f"A7  field_tilt soma 1 por partida: {a7['field_tilt_soma_1_por_partida_%']}")
    print(f"A7  soma dos alvos por partida:    {a7['soma_dos_alvos_por_partida']}")
    print(f"A6  regra 'casa vence, fora perde' (clubes): {a6['acuracia_regra_trivial_clubes_%']}%")
    print(f"A9  % de partidas de clubes até 05/2016:     {a9['%_clubes_ate_2016_05']}%")
    print(f"    temporadas de clubes pós-2016 e sua equipe dominante:")
    for k, v in a9["composicao_clubes_pos_2016"].items():
        print(f"      {k}: {v['partidas']} partidas | {v['equipe_dominante']} em {v['partidas_da_dominante']}")
    print(f"xG zerado por bloco: {sf['xg_zero_%']} | cartão vermelho: {sf['cartao_vermelho_p1_%_linhas']}%")
    print(f"\njson completo: {run.arquivo('auditoria.json')}")


if __name__ == "__main__":
    main()
