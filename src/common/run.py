"""Contexto de execução: seeds, diretório do run e manifest.json.

Regra do TCC II: todo run gera um manifest com hash do dataset, seed, versões de
biblioteca, commit e métricas. Se um número da monografia não puder ser rastreado até
um manifest, ele não entra no texto.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import caminho, raiz_projeto

BIBLIOTECAS = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "optuna", "shap", "pyyaml")


def fixar_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_arquivo(p: Path, blocos: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while bloco := fh.read(blocos):
            h.update(bloco)
    return h.hexdigest()


def versoes_bibliotecas() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    saida = {"python": platform.python_version()}
    for nome in BIBLIOTECAS:
        try:
            saida[nome] = version(nome)
        except PackageNotFoundError:
            saida[nome] = "ausente"
    return saida


def commit_git() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=raiz_projeto(), capture_output=True, text=True, timeout=15
            ).stdout.strip() or None
        except Exception:
            return None

    return {"commit": _git("rev-parse", "HEAD"), "sujo": bool(_git("status", "--porcelain"))}


@dataclass
class Run:
    """Um run de experimento com diretório próprio e manifest."""

    cfg: dict
    nome: str
    run_id: str = ""
    diretorio: Path = field(init=False)
    _inicio: datetime = field(init=False)

    def __post_init__(self) -> None:
        self._inicio = datetime.now(timezone.utc)
        if not self.run_id:
            self.run_id = f"{self._inicio.strftime('%Y%m%dT%H%M%S')}_{self.nome}"
        self.diretorio = caminho(self.cfg, "runs_dir") / self.run_id
        self.diretorio.mkdir(parents=True, exist_ok=True)
        fixar_seeds(self.cfg["seed"])

    def arquivo(self, nome: str) -> Path:
        return self.diretorio / nome

    def escrever_json(self, nome: str, dados: Any) -> Path:
        destino = self.arquivo(nome)
        with open(destino, "w", encoding="utf-8") as fh:
            json.dump(dados, fh, indent=2, ensure_ascii=False, default=str)
        return destino

    def escrever_manifest(self, **secoes: Any) -> Path:
        dataset = caminho(self.cfg, "dataset")
        manifest = {
            "run_id": self.run_id,
            "nome": self.nome,
            "inicio_utc": self._inicio.isoformat(),
            "fim_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.cfg["seed"],
            "dataset": {
                "caminho": self.cfg["caminhos"]["dataset"],
                "sha256": sha256_arquivo(dataset) if dataset.exists() else None,
                "bytes": dataset.stat().st_size if dataset.exists() else None,
            },
            "git": commit_git(),
            "bibliotecas": versoes_bibliotecas(),
            "config": self.cfg,
            **secoes,
        }
        return self.escrever_json("manifest.json", manifest)
