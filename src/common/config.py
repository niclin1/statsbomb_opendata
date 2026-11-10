"""Carregamento da configuração única do experimento.

Todo caminho do config é relativo à raiz do repositório, resolvido aqui — de modo que
os scripts funcionem independentemente do diretório de onde são chamados.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# O console do Windows usa cp1252 por padrão e falha ao imprimir os caracteres que este
# projeto usa em rótulos e mensagens (→, Δ, acentuação em algumas fontes). Como todo script
# do projeto importa este módulo, forçar UTF-8 aqui garante que os CLIs rodem sem exigir
# PYTHONIOENCODING do usuário. Um WP5 completo (6 min de execução) já foi perdido por isso:
# os resultados foram calculados e o processo morreu antes de gravá-los.
for _fluxo in (sys.stdout, sys.stderr):
    try:
        _fluxo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # fluxo redirecionado ou já configurado
        pass

CONFIG_PADRAO = "config/experiment.yaml"


def raiz_projeto() -> Path:
    """Raiz do repositório (dois níveis acima de src/common/)."""
    return Path(__file__).resolve().parents[2]


def carregar_config(caminho: str | Path | None = None) -> dict[str, Any]:
    caminho = Path(caminho) if caminho else raiz_projeto() / CONFIG_PADRAO
    if not caminho.is_absolute():
        caminho = raiz_projeto() / caminho
    with open(caminho, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_arquivo_config"] = str(caminho.relative_to(raiz_projeto()))
    return cfg


def caminho(cfg: dict, chave: str) -> Path:
    """Resolve um caminho declarado em `caminhos:` para absoluto."""
    return raiz_projeto() / cfg["caminhos"][chave]


def features(cfg: dict, conjunto: str | None = None) -> list[str]:
    """Lista de atributos de um conjunto declarado em `features.conjuntos`.

    A contagem resultante é verificada em tests/ contra o que o texto da monografia
    afirma — é o que impede a reincidência do erro "texto diz 18, dados dizem 11".
    """
    conjunto = conjunto or cfg["features"]["conjunto_principal"]
    grupos = cfg["features"]["conjuntos"][conjunto]
    nomes: list[str] = []
    for grupo in grupos:
        nomes.extend(cfg["features"][grupo])
    return nomes


def nome_competicao(cfg: dict, comp_id: int) -> str:
    entrada = cfg["competicoes"].get(int(comp_id))
    return entrada["nome"] if entrada else f"comp_{comp_id}"


def dominio_competicao(cfg: dict, comp_id: int) -> str:
    entrada = cfg["competicoes"].get(int(comp_id))
    return entrada["dominio"] if entrada else "desconhecido"
