"""Etapa tipográfica: prepara as tabelas de `outputs/tabelas/` para o build LaTeX.

Por que existe
--------------
`src/reporting/tables.py` escreve as tabelas em `outputs/tabelas/*.tex` no momento em que
cada experimento roda. Vários desses arquivos têm mais colunas do que cabe na mancha de
texto da monografia (455,0 pt medidos no template do CEFET-MG Timóteo), e transbordam a
margem em 18 a 360 pt.

Regerar as tabelas mais estreitas exigiria reexecutar os experimentos, e o conjunto de
teste já foi aberto uma única vez (`outputs/teste_aberto.json`). Reexecutar invalidaria a
estimativa não enviesada. Por isso a correção é feita aqui, depois do fato, e é puramente
tipográfica: **nenhum número é alterado**. `outputs/tabelas/` permanece como a fonte única
dos valores, exatamente como os experimentos a produziram.

O que este script faz
---------------------
1. Envolve o bloco `tabular` em `\\resizebox`, que reduz a tabela apenas quando ela é mais
   larga que a linha, e a deixa intacta quando cabe.
2. Põe em paisagem (`pdflscape`) as tabelas cuja redução deixaria a fonte abaixo de 6 pt.
   Hoje é apenas `wp2_hiperparametros.tex`, com 12 colunas: em retrato a fonte efetiva
   ficaria em 5,6 pt.
3. Acrescenta à legenda um título curto, `\\caption[curto]{completa}`, formado pela primeira
   frase da legenda. A lista de tabelas da monografia usa o título curto; sem ele, cada
   entrada da lista reproduziria a legenda inteira, com as notas de leitura.
4. Troca a marca de célula sem valor por `n/a`. A origem usa duas marcas para o mesmo caso,
   o travessão e `--`; o travessão é vetado pelo regimento de escrita (`vocabulário.md`).
5. Aplica as correções de texto listadas em `SUBSTITUICOES`, em legendas e cabeçalhos.
   Cada trecho precisa ocorrer exatamente uma vez na origem; se a origem mudar, o script
   aborta em vez de deixar a correção cair em silêncio.

6. Escreve os intervalos de confiança como `41,76 a 50,59` no lugar de `[41.76; 50.59]`.
   Em célula que traz a estimativa e o intervalo, o intervalo vai entre parênteses.
7. Usa vírgula como separador decimal nas células, como no restante do texto em português.
   A origem usa ponto porque é escrita direto do `pandas`. Só o ponto entre dois dígitos é
   trocado; datas, nomes e legendas ficam como estão.

Nenhum dos passos altera valor numérico: os dígitos de cada tabela de build são os mesmos da
origem, na mesma ordem.

É idempotente: rodar duas vezes produz o mesmo resultado.

Uso
---
    python -m src.reporting.formatar_tabelas_tex
"""
from __future__ import annotations

import re
from pathlib import Path

_INTERVALO = re.compile(r"\[([^\];]+); ([^\]]+)\]")
_PONTO_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")

ORIGEM = Path("outputs/tabelas")
DESTINO = Path("monografia-template1/tabelas")

# Tabelas que vão para paisagem. A fonte efetiva de uma tabela reduzida é
# largura_da_mancha x corpo_base / largura_natural, e não depende do corpo escolhido.
# Abaixo de 6 pt a leitura em papel deixa de ser confiável.
PAISAGEM = {"wp2_hiperparametros.tex"}

# Marcas de célula sem valor na origem, unificadas em uma só no build.
_SEM_VALOR = {"—", "--"}
_MARCA_SEM_VALOR = "n/a"

# Correções de texto em legendas e cabeçalhos: arquivo -> [(trecho exato, substituto)].
SUBSTITUICOES: dict[str, list[tuple[str, str]]] = {
    # A legenda dizia "maximizando F1 macro", mas a tabela também traz as configurações
    # otimizadas para acurácia (coluna "Otimizado para"): são quatro estudos de 50 tentativas.
    "wp2_hiperparametros.tex": [
        (
            "por conjunto de atributos, maximizando F1 macro nas dobras de validação.",
            "por conjunto de atributos e por métrica de otimização (F1 macro ou acurácia), "
            "nas dobras de validação.",
        ),
    ],
    # Travessão e ponto e vírgula em texto corrido de legenda.
    "wp6_calibracao.tex": [
        (" — positivo indica", ". Valor positivo indica"),
        ("(menor é melhor); \\emph{ECE} é", "(menor é melhor). \\emph{ECE} é"),
        (
            "confiança e acerto; \\emph{excesso de confiança} é",
            "confiança e acerto. \\emph{Excesso de confiança} é",
        ),
    ],
    # Ponto e vírgula em texto corrido de legenda: cada oração vira uma frase.
    # "Baseline" vira "regra de referência", termo usado no texto da monografia, e o
    # cabeçalho troca os nomes de coluna do pandas pelos nomes das métricas no texto.
    "baselines.tex": [
        (
            "Baselines por partição (valores em \\%). O baseline de classe majoritária",
            "Regras de referência por partição (valores em \\%). A regra de classe majoritária",
        ),
        (
            "base espelhada; o de mando de campo é a regra trivial",
            "base espelhada. A regra de mando de campo é",
        ),
        (
            "Partição & Baseline & Partidas & acuracia & f1\\_Derrota & f1\\_Empate"
            " & f1\\_Vitória & f1\\_macro",
            "Partição & Regra de referência & Partidas & Acurácia & F1 Derrota & F1 Empate"
            " & F1 Vitória & F1 macro",
        ),
    ],
    # Dois-pontos explicativos em legenda e nomes de coluna do pandas no cabeçalho.
    # "treino ∪ validação" vira "janela de ajuste", termo usado no texto.
    "particoes.tex": [
        (
            "Domínio primário: cinco grandes ligas europeias, temporada 2015/16, com corte",
            "O domínio primário é a temporada 2015/16 das cinco grandes ligas europeias, com corte",
        ),
    ],
    "diag_atributos_adversario.tex": [
        ("em treino $\\cup$ validação.", "na janela de ajuste (treino e validação)."),
        (
            "omitido por ser redundante: as duas equipes somam 1,0 por construção.",
            "omitido porque os valores das duas equipes somam 1,0 por construção.",
        ),
        (
            "Conjunto & Atributos & acuracia & f1\\_macro & f1\\_Derrota & f1\\_Empate"
            " & f1\\_Vitória & log\\_loss",
            "Conjunto & Atributos & Acurácia & F1 macro & F1 Derrota & F1 Empate"
            " & F1 Vitória & Log-Loss",
        ),
    ],
    "wp3_teste.tex": [
        ("Desempenho na partição teste (340", "Desempenho na partição de teste (340"),
    ],
    "wp5_progressao.tex": [
        ("em treino $\\cup$ validação.", "na janela de ajuste (treino e validação)."),
        ("Acur. (default) & F1 macro (default)", "Acur. (padrão) & F1 macro (padrão)"),
    ],
    # "TCC I" é o nome do documento da etapa anterior. O texto da monografia chama essa etapa
    # de "etapa anterior deste trabalho", e as tabelas passam a usar a mesma forma. O rótulo de
    # linha usa a forma curta "etapa anterior" para caber na coluna. Os geradores em
    # src/modeling/efeito_composicao.py já escrevem a forma nova, mas outputs/tabelas/ guarda
    # a forma antiga porque eles não podem ser reexecutados (o teste já foi aberto). Se as
    # tabelas forem regeradas, estas entradas deixam de casar e o script aborta: removê-las.
    "efeito_composicao.tex": [
        ("índice de linha; a partição correta", "índice de linha. A partição correta"),
        (
            "A partição do TCC I foi reconstruída",
            "A partição usada na etapa anterior deste trabalho foi reconstruída",
        ),
        ("Teste do TCC I (reconstruído)", "Teste da etapa anterior (reconstruído)"),
    ],
    "efeito_composicao_decomposicao.tex": [
        (
            "Decomposição do conjunto de teste do TCC I por competição.",
            "Decomposição por competição do conjunto de teste da etapa anterior deste trabalho.",
        ),
    ],
    "wp2_comparativo.tex": [
        ("exceto Log-Loss; \\emph{sd} é", "exceto Log-Loss. A coluna \\emph{sd} é"),
        (
            ", em treino $\\cup$ validação (o teste não participa).",
            " na janela de ajuste (treino e validação), sem o conjunto de teste.",
        ),
    ],
    "wp5_incrementos.tex": [
        ("anterior; \\emph{acum.} é", "anterior. \\emph{acum.} é"),
        (
            "Δ acuracia (bloco) & dobras acuracia & Δ acuracia (acum. vs 45 min)"
            " & Δ f1\\_macro (bloco) & dobras f1\\_macro & Δ f1\\_macro (acum. vs 45 min)",
            "Δ acurácia (bloco) & Dobras (acurácia) & Δ acurácia (acum. sobre 45 min)"
            " & Δ F1 macro (bloco) & Dobras (F1 macro) & Δ F1 macro (acum. sobre 45 min)",
        ),
    ],
    # Travessão em cabeçalho de coluna.
    "wp4_simetria_empate.tex": [
        ("Dif. média — empates", "Dif. média (empates)"),
        ("Dif. média — decididas", "Dif. média (decididas)"),
    ],
}

_ABRE = (
    "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%"
)
_MARCA = "% Ajuste tipográfico aplicado por src/reporting/formatar_tabelas_tex.py."


def _titulo_curto(linha: str) -> str:
    """`\\caption{A. B.}` vira `\\caption[A]{A. B.}`. Nenhum número é alterado."""
    prefixo = "\\caption{"
    if not linha.startswith(prefixo) or not linha.endswith("}"):
        return linha
    legenda = linha[len(prefixo):-1]
    # ". " separa frases; "2.635" não tem espaço depois do ponto e não é cortado.
    curto = legenda.split(". ")[0].rstrip(".")
    return f"\\caption[{curto}]{{{legenda}}}"


def _celula(c: str) -> str:
    """Marca de valor ausente, notação de intervalo e vírgula decimal de uma célula."""
    if c.strip() in _SEM_VALOR:
        return _MARCA_SEM_VALOR
    if _INTERVALO.fullmatch(c.strip()):  # coluna só com o intervalo
        c = _INTERVALO.sub(r"\1 a \2", c)
    else:  # estimativa seguida do intervalo
        c = _INTERVALO.sub(r"(\1 a \2)", c)
    return _PONTO_DECIMAL.sub(",", c)


def _celulas_sem_valor(linha: str) -> str:
    """Aplica `_celula` a cada célula de uma linha do corpo da tabela."""
    fim = " \\\\"
    if not linha.endswith(fim):
        return linha
    celulas = linha[: -len(fim)].split(" & ")
    return " & ".join(_celula(c) for c in celulas) + fim


def _ajustar(texto: str, paisagem: bool, nome: str = "") -> str:
    if _MARCA in texto:  # idempotência
        return texto
    for antigo, novo in SUBSTITUICOES.get(nome, []):
        if texto.count(antigo) != 1:
            raise ValueError(
                f"{nome}: esperava 1 ocorrência de {antigo!r}, achei {texto.count(antigo)}. "
                "A tabela de origem mudou; revise SUBSTITUICOES."
            )
        texto = texto.replace(antigo, novo)
    linhas = texto.split("\n")
    saida: list[str] = []
    dentro_do_tabular = False
    for linha in linhas:
        if dentro_do_tabular and not linha.startswith("\\end{tabular}"):
            saida.append(_celulas_sem_valor(linha))
            continue
        if linha.startswith("\\begin{tabular}"):
            dentro_do_tabular = True
            saida.append(_ABRE)
            saida.append(linha)
        elif linha.startswith("\\end{tabular}"):
            dentro_do_tabular = False
            saida.append(linha + "%")
            saida.append("}")
        elif linha.startswith("\\caption{"):
            saida.append(_titulo_curto(linha))
        elif linha.startswith("\\begin{table}") and paisagem:
            saida.append("\\begin{landscape}")
            saida.append(linha)
        elif linha.startswith("\\end{table}") and paisagem:
            saida.append(linha)
            saida.append("\\end{landscape}")
        else:
            saida.append(linha)
    cabecalho = [
        _MARCA,
        "% Fonte dos números: " + "outputs/tabelas/ (não editar nem aqui nem lá).",
    ]
    return "\n".join(cabecalho + saida)


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for origem in sorted(ORIGEM.glob("*.tex")):
        alvo = DESTINO / origem.name
        alvo.write_text(
            _ajustar(
                origem.read_text(encoding="utf-8"), origem.name in PAISAGEM, origem.name
            ),
            encoding="utf-8",
        )
        marca = " (paisagem)" if origem.name in PAISAGEM else ""
        print(f"{origem} -> {alvo}{marca}")


if __name__ == "__main__":
    main()
