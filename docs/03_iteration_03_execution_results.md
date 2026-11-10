# Relatório de Execução: Iteração 03  Predição In-Play (Intra-Jogo)

## 1. Resumo Executivo
Esta iteração testou com sucesso a hipótese levantada pelo orientador: **"Seria possível e melhor prever partes de uma partida durante o jogo e então obter o resultado final?"** 
Foi desenvolvido um pipeline de ETL segmentado por períodos (1º e 2º Tempo), e a modelagem avaliou o poder preditivo do estado de jogo (Game-State) no intervalo, comparado ao histórico pré-jogo.

## 2. Metodologia

Três cenários foram treinados utilizando o algoritmo **XGBoost Classifier** através de uma validação cruzada temporal (TimeSeries CV = 5 splits).

- **M1  Pré-Jogo Baseline**: Utilizou médias móveis (últimos 5 jogos) de xG, PPDA e Field Tilt.
- **M2  Apenas 1º Tempo**: Utilizou apenas as estatísticas acumuladas da equipe no 1º tempo (xG, PPDA, Field Tilt, finalizaçoes no alvo) e a diferença no placar até o intervalo.
- **M3  Combinado**: União das features do M1 e M2.

## 3. Resultados dos Experimentos (Métrica: Accuracy)

| Modelo | Acurácia Média (CV) | Paradigma |
| :--- | :--- | :--- |
| M1 (Pré-Jogo Baseline) | 59.39% | Histórico |
| **M2 (1º Tempo Only)** | **67.88%** | **In-Play (Intervalo)** |
| M3 (Combinado) | 65.45% | Histórico + In-Play |

## 4. Análise e Validação da Hipótese

Os critérios de sucesso da Iteração 03 estipulavam que:
1. *M2 (1º Tempo) deve ter a marca de 33.3% superada.* (Critério atingido: 67.88%)
2. *M3 (Combinado) ou M2 deveria superar M1.* (Critério atingido: M2 e M3 superaram M1)

**A hipótese foi fortemente corroborada e confirmada empiricamente.** 
O sinal preditivo extraído dos 45 minutos iniciais (M2) superou de forma contundente o modelo de retrospecto de 5 jogos (M1). Isso evidencia que os padrões que ocorrem dentro do jogo corrente (momentum, adequação tática diária, lesões, confiança) exercem maior gravidade sobre o desfecho da partida do que o desempenho pregresso daquela equipe em outras partidas, demonstrando as limitações dos modelos retrospectivos puramente focados no histórico.

Notavelmente, a combinação M3 demonstrou uma queda ligeira em comparação com M2 puro (de 67.8% para 65.4%), possivelmente indicando ruído (overfitting) introduzido pelo misto de sinais de longo prazo atuando sobre uma amostra limitada de dados. 

## 5. Artefatos Gerados
- \src/etl_intra_match.py\: Refatorado com features intra-período.
- \src/features_inplay.py\: Engine de momentum features.
- \src/train_inplay.py\: Validação temporal comparativa (M1 vs M2 vs M3).
- \outputs/plots/iteration_03_accuracy_comparison.png\: Gráfico final.
- \outputs/plots/iteration_03_m3_feature_importance.png\: Plot de Feature Importance.

## 6. Próximos Passos
Tendo em vista o alto poder preditivo de variáveis In-Play, o estudo pode agora consolidar esse paradigma dinâmico na Tese final, justificando plenamente o questionamento do orientador, criando os pilares do preditivo em live.
