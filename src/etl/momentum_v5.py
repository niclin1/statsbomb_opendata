"""ETL v5 — estende o v4 com os blocos do **segundo tempo** (WP5).

Motivação: a banca perguntou por que os dados do segundo tempo não são usados. Responder exige
o experimento, não o argumento. Este ETL produz a base que permite comparar, sob o mesmo
protocolo do WP1, apenas-1º-tempo contra 1º+2º tempo — e, mais útil, a progressão bloco a bloco.

**Decisão de projeto: as funções de cálculo são importadas de `src/etl_momentum_v4.py`, não
reescritas.** Assim os atributos do primeiro tempo são, por construção, idênticos aos da base
atual, e a comparação entre as variantes fica pareada. A igualdade é verificada em
`verificar_reproducao()` antes de qualquer uso.

Blocos:
    1º tempo (period 1): 0–15, 15–30, 30–45   — idênticos ao v4
    2º tempo (period 2): 45–60, 60–75, 75–90  — mesma lógica, deslocada em 45 min

Uso:
    python -m src.etl.momentum_v5 --amostra 20   # ensaio + verificação de reprodução
    python -m src.etl.momentum_v5                # base completa
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from ..common.config import caminho, carregar_config, raiz_projeto
from ..etl_momentum_v4 import (  # noqa: E402  — o import instala o cache do StatsBomb
    calculate_ppda_segment,
    calculate_xg_segment,
    get_field_tilt_segment,
    is_red_card,
)

from statsbombpy import sb  # noqa: E402

# (nome do bloco, período, minuto inicial, minuto final ou None para "até o fim")
BLOCOS = [
    ("0_15", 1, 0, 15),
    ("15_30", 1, 15, 30),
    ("30_45", 1, 30, None),
    ("45_60", 2, 45, 60),
    ("60_75", 2, 60, 75),
    ("75_90", 2, 75, None),
]
BLOCOS_1T = [b for b in BLOCOS if b[1] == 1]
BLOCOS_2T = [b for b in BLOCOS if b[1] == 2]

COMPETICOES = [
    "La Liga", "Premier League", "Serie A", "Ligue 1", "1. Bundesliga",
    "FIFA World Cup", "UEFA Euro", "Copa America", "Indian Super league",
]


def _fatiar(eventos: pd.DataFrame, periodo: int, ini: int, fim: int | None) -> pd.DataFrame:
    df = eventos[eventos["period"] == periodo]
    df = df[df["minute"] >= ini]
    if fim is not None:
        df = df[df["minute"] < fim]
    return df.copy()


def processar_partida(detalhes: pd.Series, comp_id: int, season_id: int) -> list[dict]:
    match_id = detalhes["match_id"]
    casa, fora = detalhes["home_team"], detalhes["away_team"]

    try:
        eventos = sb.events(match_id=match_id)
    except Exception:
        return []
    if eventos.empty or "period" not in eventos.columns or "minute" not in eventos.columns:
        return []

    metricas = {casa: {}, fora: {}}
    for nome, periodo, ini, fim in BLOCOS:
        seg = _fatiar(eventos, periodo, ini, fim)
        ft = get_field_tilt_segment(seg)
        for t in (casa, fora):
            metricas[t][f"xg_{nome}"] = calculate_xg_segment(seg, t)
            metricas[t][f"ppda_{nome}"] = calculate_ppda_segment(seg, t)
            metricas[t][f"field_tilt_{nome}"] = ft.get(t, 0.0)

    # cartões vermelhos por tempo
    reds = {}
    for periodo, sufixo in ((1, "p1"), (2, "p2")):
        dfp = eventos[eventos["period"] == periodo]
        if dfp.empty:
            reds[f"casa_{sufixo}"] = reds[f"fora_{sufixo}"] = 0
            continue
        marcados = dfp[dfp.apply(is_red_card, axis=1)]
        reds[f"casa_{sufixo}"] = int((marcados["team"] == casa).sum())
        reds[f"fora_{sufixo}"] = int((marcados["team"] == fora).sum())

    if detalhes["home_score"] > detalhes["away_score"]:
        alvo_casa, alvo_fora = 2, 0
    elif detalhes["home_score"] < detalhes["away_score"]:
        alvo_casa, alvo_fora = 0, 2
    else:
        alvo_casa = alvo_fora = 1

    tem_2t = bool((eventos["period"] == 2).any())
    linhas = []
    for equipe, adv, is_home, alvo, meu, dele in [
        (casa, fora, 1, alvo_casa, "casa", "fora"),
        (fora, casa, 0, alvo_fora, "fora", "casa"),
    ]:
        linha = {
            "match_id": match_id,
            "match_date": detalhes["match_date"],
            "competition_id": comp_id,
            "season_id": season_id,
            "team_name": equipe,
            "opponent_name": adv,
            "is_home": is_home,
            "target_multiclass": alvo,
            "team_red_cards_p1": reds[f"{meu}_p1"],
            "opp_red_cards_p1": reds[f"{dele}_p1"],
            "team_red_cards_p2": reds[f"{meu}_p2"],
            "opp_red_cards_p2": reds[f"{dele}_p2"],
            "tem_segundo_tempo": tem_2t,
        }
        linha.update(metricas[equipe])
        linhas.append(linha)
    return linhas


def executar(limite: int | None = None, amostra: int | None = None) -> pd.DataFrame:
    comps = sb.competitions()
    alvo = comps[
        comps["competition_name"].isin(COMPETICOES)
        & (comps["season_name"].str.startswith("202") | comps["season_name"].str.startswith("201"))
    ]
    print(f"{len(alvo)} temporadas alvo")

    linhas, n = [], 0
    t0 = time.time()
    for _, comp in alvo.iterrows():
        try:
            partidas = sb.matches(competition_id=comp["competition_id"], season_id=comp["season_id"])
        except Exception:
            continue
        for _, partida in partidas.iterrows():
            linhas.extend(processar_partida(partida, comp["competition_id"], comp["season_id"]))
            n += 1
            if n % 200 == 0:
                print(f"  {n} partidas | {(time.time()-t0)/60:.1f} min")
            if amostra and n >= amostra:
                return pd.DataFrame(linhas)
        if limite and n >= limite:
            break
    print(f"total: {n} partidas em {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(linhas)


def verificar_reproducao(novo: pd.DataFrame, cfg: dict) -> dict:
    """O 1º tempo do v5 tem de ser idêntico ao da base atual, partida a partida.

    É a garantia de que a comparação do WP5 é pareada: qualquer diferença aqui significaria
    que as duas variantes não são o mesmo experimento com atributos a mais.
    """
    antigo = pd.read_csv(caminho(cfg, "dataset"))
    cols_1t = [
        f"{m}_{b[0]}" for b in BLOCOS_1T for m in ("xg", "ppda", "field_tilt")
    ] + ["team_red_cards_p1", "opp_red_cards_p1", "target_multiclass", "is_home"]

    chave = ["match_id", "team_name"]
    a = antigo.set_index(chave)[cols_1t].sort_index()
    b = novo.set_index(chave)[cols_1t].sort_index()
    comuns = a.index.intersection(b.index)
    a, b = a.loc[comuns], b.loc[comuns]

    divergencias = {}
    for c in cols_1t:
        if pd.api.types.is_numeric_dtype(a[c]):
            dif = (a[c] - b[c]).abs()
            n_dif = int((dif > 1e-9).sum())
        else:
            n_dif = int((a[c] != b[c]).sum())
        if n_dif:
            divergencias[c] = n_dif
    return {"linhas_comparadas": int(len(comuns)), "divergencias": divergencias}


def main() -> None:
    ap = argparse.ArgumentParser(description="ETL v5 — 1º e 2º tempos")
    ap.add_argument("--amostra", type=int, default=None, help="processa só N partidas (ensaio)")
    args = ap.parse_args()

    assert Path.cwd() == raiz_projeto(), (
        f"rode a partir da raiz do repositório ({raiz_projeto()}) — o cache do StatsBomb "
        "é resolvido por caminho relativo"
    )
    cfg = carregar_config()
    df = executar(amostra=args.amostra)
    print(f"\nlinhas: {len(df)} | partidas: {df.match_id.nunique()}")
    print(f"partidas sem 2º tempo: {int((~df.tem_segundo_tempo).sum()) // 2}")

    check = verificar_reproducao(df, cfg)
    print(f"\n== verificação de reprodução do 1º tempo ==")
    print(f"  linhas comparadas: {check['linhas_comparadas']}")
    print(f"  divergências: {check['divergencias'] or 'NENHUMA — reprodução exata'}")
    if check["divergencias"]:
        raise SystemExit("ETL v5 não reproduz o 1º tempo do v4; comparação do WP5 seria inválida.")

    if args.amostra:
        print("\n(ensaio — nada gravado)")
        return

    destino = caminho(cfg, "processed_dir")
    destino.mkdir(parents=True, exist_ok=True)
    arq = destino / f"momentum_v5_{time.strftime('%Y%m%d')}.parquet"
    df.to_parquet(arq, index=False)
    print(f"\ngravado: {arq.relative_to(raiz_projeto())} ({arq.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
