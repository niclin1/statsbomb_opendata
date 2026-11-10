"""Geração de tabelas .tex a partir de DataFrames.

Regra do TCC II: nenhum número entra na monografia manualmente. As tabelas do texto são
`\\input{}` destes arquivos, gerados a partir de `outputs/runs/`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}


def _escapar(valor: object) -> str:
    texto = "--" if valor is None or (isinstance(valor, float) and pd.isna(valor)) else str(valor)
    for de, para in _ESCAPES.items():
        texto = texto.replace(de, para)
    return texto


def escrever_latex(
    df: pd.DataFrame,
    destino: str | Path,
    legenda: str,
    rotulo: str,
    alinhamento: str | None = None,
) -> Path:
    """Escreve uma tabela booktabs. Colunas numéricas alinhadas à direita."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if alinhamento is None:
        alinhamento = "".join(
            "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns
        )

    linhas = [
        "% Gerado automaticamente por src/reporting/tables.py — não editar à mão.",
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{legenda}}}",
        f"\\label{{{rotulo}}}",
        r"\small",
        f"\\begin{{tabular}}{{{alinhamento}}}",
        r"\toprule",
        " & ".join(_escapar(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, linha in df.iterrows():
        linhas.append(" & ".join(_escapar(v) for v in linha) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino
