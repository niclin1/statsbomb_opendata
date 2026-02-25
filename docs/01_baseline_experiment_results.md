# Análise do Experimento Inicial: Modelo Baseline (Copa do Mundo 2018)

Este documento detalha os resultados da primeira iteração do modelo preditivo, utilizando dados da Copa do Mundo de 2018.

## Metodologia
- **Dados**: Campeonato curto (Copa do Mundo), totalizando 64 partidas.
- **Features**: Médias móveis (Rolling Averages) de métricas táticas (Field Tilt, Convex Hull, Zone 14 Passes) das últimas 5 partidas de cada time.
- **Divisão**: Split cronológico (75% Treino / 25% Teste).

## Resultados Obtidos
A execução do script `src/train_baseline.py` gerou os seguintes resultados:

- **Total de Partidas no Dataset**: 64 partidas.
- **Partidas Válidas para Treino**: Apenas 16 partidas (32 linhas de dados).
    - O filtro de "histórico mínimo de 5 jogos" descartou 48 partidas iniciais (Fase de Grupos inteira e parte do mata-mata).
- **Amostras no Conjunto de Teste**: 4 partidas.
- **Acurácia do Modelo**: 0.0% (Zero).
- **Matriz de Confusão**: O modelo falhou em generalizar, prevendo incorretamente todas as classes.

## Discussão e Conclusão (Para o TCC)

A análise dos resultados aponta uma falha estrutural na escolha dos dados para este tipo de modelagem temporal.

> "Percebi que a tentativa inicial de treinar um modelo usando apenas os dados de um torneio curto resultou em uma base de dados extremamente reduzida. Ao aplicar janelas deslizantes para calcular médias móveis em um contexto de poucos jogos, a maior parte das informações acaba sendo consumida para gerar o histórico, sobrando poucas amostras para o treinamento real. Isso explica a performance inconsistente observada e me leva à conclusão de que precisamos migrar para um campeonato de pontos corridos mais extenso para obter o volume estatístico necessário para um aprendizado de máquina robusto."

### Impacto da Janela Deslizante em Torneios Curtos
Em uma Copa do Mundo, as seleções jogam no máximo 7 partidas. Ao exigir um histórico prévio de 5 jogos para calcular a performance média de um time:
1. Os jogos 1 a 5 de cada seleção servem **apenas** para construir o histórico.
2. Somente a partir do jogo 6 (Semifinais/Final) temos dados de entrada completos para o modelo.
3. Isso reduz o dataset útil de 64 partidas para menos de 10 jogos "treináveis", o que é insuficiente para qualquer algoritmo de Machine Learning (Logistic Regression ou Deep Learning) encontrar padrões estatísticos significativos.

## Diagnóstico Sistêmico e Análise de Falhas da Fase Piloto

A implementação inicial do projeto Anti Gravity, embora bem-sucedida na extração algorítmica de métricas geométricas complexas como Field Tilt e Convex Hull para a final da Copa do Mundo FIFA, revelou uma fragilidade estrutural crítica na arquitetura de dados e na formulação do problema de modelagem. O resultado obtido — um dataset final de apenas 32 linhas e uma acurácia de modelo de 0% — não representa meramente um desempenho subótimo, mas sim um erro fundamental na definição da unidade de análise estatística. Este relatório estabelece as bases teóricas e práticas para a correção deste cenário, detalhando a transição de uma análise descritiva de partida única para um sistema preditivo robusto alimentado por dados históricos massivos.

### A Patologia da Amostra Minúscula (N=32)

O fenômeno observado, onde o dataset resultante contém apenas 32 observações, é sintomático de uma agregação de dados realizada ao nível do jogador (player-level aggregation) restrita a um único evento (a partida final). Em uma partida de futebol profissional, cada equipe inicia com 11 jogadores e pode realizar até 5 ou 6 substituições, totalizando aproximadamente 16 participantes por lado. A soma dos atletas de ambas as equipes (Argentina e França) resulta precisamente no volume de dados relatado.

Embora esta granularidade seja valiosa para a avaliação de desempenho individual (scouting), ela é matematicamente inadequada para a modelagem preditiva de resultados de partidas. Modelos de aprendizado de máquina supervisionados, como Random Forest ou Gradient Boosting, operam identificando padrões de covariância entre variáveis independentes (features) e uma variável dependente (target) através de múltiplas instâncias do evento que se deseja prever. Ao tentar prever o "Vencedor da Partida" usando jogadores como unidade amostral dentro de um único jogo, o modelo enfrenta um paradoxo lógico:

Se o target for o resultado do time, todos os jogadores da Argentina teriam target=1 (Vitória) e todos da França target=0 (Derrota/Empate).

Qualquer divisão de treino e teste (train-test split) realizada dentro desse universo de 32 linhas resultaria em vazamento de dados (data leakage) ou em classes não representadas. Por exemplo, se o conjunto de treino contiver apenas jogadores da Argentina, o modelo aprenderá que "todos ganham", falhando catastroficamente ao encontrar um jogador da França no conjunto de teste.

A acurácia de 0% sugere que o modelo foi incapaz de generalizar ou que a métrica de avaliação foi aplicada a um conjunto de dados onde a classe predita não existia.

Para que o sistema Anti Gravity atinja seus objetivos preditivos, é imperativo redefinir a unidade de análise para a Equipe-Partida (Team-Match). Sob esta nova ótica, cada linha do dataset deve representar o desempenho agregado de uma equipe em um jogo específico. Consequentemente, para treinar um modelo viável, o volume de dados deve ser expandido de 1 jogo para centenas ou milhares de jogos históricos.

### Limitações da Abordagem Descritiva vs. Preditiva

A fase piloto demonstrou competência na engenharia de métricas descritivas. O cálculo correto do Convex Hull e do Field Tilt indica que o pipeline de processamento geométrico dos eventos (event_freeze_frame ou coordenadas de passes) está funcional. No entanto, métricas descritivas explicam o passado (post-mortem), enquanto o objetivo do Anti Gravity é antecipar resultados futuros. A falha em obter acurácia reside na ausência de variância histórica.

## Próximos Passos Recomendados
Para viabilizar o projeto, é necessário alterar a fonte de dados para uma liga regular (ex: La Liga, Premier League), onde cada time disputa 38 rodadas. Nesse cenário, perder as 5 primeiras rodadas para "aquecimento" ainda deixaria 33 rodadas completas de dados para treino e validação, multiplicadas por 20 times, totalizando centenas de amostras válidas.
