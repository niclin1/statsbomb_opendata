"""
Gera as versoes em PNG dos dois diagramas da metodologia:
  - fig_metodologia_fluxo.png : desenvolvimento ITERATIVO do modelo preditivo
  - fig_base_dados.png        : do evento de futebol (texto/JSON) ao conjunto de atributos

As versoes equivalentes em TikZ estao embutidas em monografia-template1/monografia.tex
(alterne com \\usetikztrue / \\usetikzfalse). Saida direto na pasta fig/ do template.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "monografia-template1", "fig"))
os.makedirs(OUT, exist_ok=True)

EDGE = "#333333"
IO_FILL = "#e6e6e6"
IT_FILL = "#e8f0fe"
JSON_FILL = "#f2f2f2"


def _box(ax, cx, cy, w, h, text, fill="white", fontsize=10, family="sans-serif",
         align="center", pad_left=0.18):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=EDGE, facecolor=fill, zorder=2))
    if align == "left":
        ax.text(cx - w / 2 + pad_left, cy, text, ha="left", va="center",
                fontsize=fontsize, family=family, color="black", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                family=family, color="black", zorder=3)


def _arrow(ax, x1, y1, x2, y2, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.5, color=("#777777" if dashed else EDGE),
        linestyle=("--" if dashed else "-"),
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def fluxo_iterativo():
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 10)
    ax.axis("off")
    cy = 6.4
    iw, ih = 3.0, 2.3   # io boxes
    tw, th = 3.5, 3.1   # iteration cards

    nodes = [
        ("io", "Base de dados\n+ métricas"),
        ("it", "Iteração 1\nProva de conceito\n(torneio curto)"),
        ("it", "Iteração 2\nModelo pré-jogo\n+ equipe-partida"),
        ("it", "Iteração 3\nDados intrajogo\n+ 1º tempo"),
        ("it", "Iteração 4\nSegmentação e\nmulticlasse\n+ blocos de 15 min"),
        ("it", "Iteração 5\nOtimização\n+ Optuna, + SHAP"),
        ("io", "Modelo final\n+ análise de\nresultados"),
    ]
    centers = [2.0, 5.7, 9.2, 12.7, 16.2, 19.7, 23.6]
    half = []
    for (kind, text), cx in zip(nodes, centers):
        if kind == "io":
            _box(ax, cx, cy, iw, ih, text, fill=IO_FILL, fontsize=9.5)
            half.append(iw / 2)
        else:
            # bold first line (iteration label)
            _box(ax, cx, cy, tw, th, text, fill=IT_FILL, fontsize=9)
            half.append(tw / 2)
    for i in range(len(centers) - 1):
        _arrow(ax, centers[i] + half[i], cy, centers[i + 1] - half[i + 1], cy)

    # nota da progressao iterativa (sem seta de retorno, para nao sugerir volta a iteracao 1)
    i1, i5 = centers[1], centers[5]
    ax.text((i1 + i5) / 2, cy - th / 2 - 0.9,
            "cada iteração parte do diagnóstico da anterior",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444444")

    fig.tight_layout()
    path = os.path.join(OUT, "fig_metodologia_fluxo.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("salvo:", path)


def evento_para_atributos():
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 11)
    ax.axis("off")
    cy = 6.0

    json_text = (
        '{\n'
        '  "minute": 23, "second": 47,\n'
        '  "team": "Argentina",\n'
        '  "player": "L. Messi",\n'
        '  "type": "Shot",\n'
        '  "location": [104.2, 37.9],\n'
        '  "shot": { "outcome": "Goal",\n'
        '            "statsbomb_xg": 0.31 }\n'
        '}'
    )
    _box(ax, 4.6, cy, 8.8, 6.0, json_text, fill=JSON_FILL,
         fontsize=10, family="monospace", align="left", pad_left=0.35)

    _box(ax, 12.7, cy, 4.4, 6.0,
         "Cálculo de\nmétricas táticas\n\nxG, PPDA,\nField Tilt,\nConvex Hull, ...",
         fontsize=11)

    rep_text = (
        "Representação (varia por iteração)\n\n"
        "It. 1:  por jogador\n"
        "It. 2–3:  por equipe-partida\n"
        "It. 4–5:  por blocos de 15 min\n\n"
        "Configuração final (It. 4–5):\n"
        "vetor de 18 atributos\n"
        "(xG, PPDA, Field Tilt e\n"
        "cartão vermelho por bloco)"
    )
    _box(ax, 20.6, cy, 6.4, 6.0, rep_text, fill=IT_FILL,
         fontsize=9.5, align="left", pad_left=0.3)

    _arrow(ax, 9.05, cy, 10.45, cy)
    _arrow(ax, 14.95, cy, 17.35, cy)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_base_dados.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("salvo:", path)


if __name__ == "__main__":
    fluxo_iterativo()
    evento_para_atributos()
    print("OK - diagramas gerados em", OUT)
