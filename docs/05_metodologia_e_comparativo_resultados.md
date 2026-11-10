# Resumo Comparativo: Metodologia e Resultados (Abordagem Multiclasse)

Este documento centraliza a comparação das métricas e sumariza as metodologias aplicadas na previsão de resultados de partidas de futebol. A análise descreve o comportamento dos modelos preditivos após a adequação do escopo do problema para classificação multiclasse.

## 1. Descrição Metodológica

A variável alvo (`target`) foi modelada com três classes discretas, representando a totalidade dos resultados possíveis no futebol esportivo regular: Vitória (2), Empate (1) e Derrota (0). 

A abordagem altera o *baseline* de acerto aleatório, que oscila de ~50% (problemas dicotômicos) para matematicamente ~33.3%. O ajuste assegura que as avaliações de desempenho derivadas dos modelos captem a nuance real do esporte, onde a previsão de empates é uma tarefa naturalmente acompanhada de maior variância e ruído.

Os algoritmos selecionados para os experimentos incluíram *Random Forest* (modelo de referência não-linear) e *XGBoost* (modelo baseado em árvores e *Gradient Boosting*). A função objetivo utilizada nas fases avançadas do XGBoost foi configurada como `multi:softprob` com `num_class=3`. As métricas principais para todas as avaliações empíricas foram orientadas para Acurácia Global, *F1-Score (Macro)* para ponderar o desbalanceamento inerente entre vitórias/derrotas e empates, e *Log-Loss* multiclasse.

## 2. Comparativo de Modelos: Pré-Jogo (Dados Históricos)

Neste estágio, os modelos foram treinados sobre estatísticas das últimas 5 partidas das equipes (`modeling_data.csv`). Uma divisão cronológica foi estabelecida (Treino: 2008 a 2016; Teste: 2016 a 2022). 

Os resultados obtidos sobre os dados de teste registraram as seguintes métricas:

| Abordagem                      | Acurácia | F1-Score (Macro) | Log-Loss | Observação |
|--------------------------------|----------|-------------------|----------|------------|
| Algoritmo Aleatório (Lógico)   | ~33.3%   | -                 | -        | Limiar de referência básico. |
| Random Forest                  | 65.22%   | 49.41%            | 0.9286   | Performance de generalização da literatura. |
| XGBoost (Challenger)           | 61.14%   | 49.14%            | 0.9895   | Menor acurácia pontual em relação ao Random Forest. |

*Nota Funcional:* A variação e a estabilidade nas previsões evidenciam aprendizado sistemático em relação ao *baseline* matemático, operando acima do limiar de 33.3%. A capacidade dos modelos de extrair sinal e distinguir entre classes é verificada, embora o *F1-Score (Macro)* abaixo da acurácia global indique dificuldade na classificação da classe minoritária ou ruidosa (Empate).

## 3. Comparativo de Modelos: In-Play (Validação Cruzada Temporal)

Em seguida, o controle de dados intrapartida (`inplay_modeling_data.csv`) foi explorado mediante o fracionamento do jogo no primeiro tempo. Os dados são originados do *pipeline* que processa ações, coordenadas interativas, métricas de pressão e *Expected Goals (xG)* agregadas pelos períodos de jogo, simulando modelos de previsão que atualizam no intervalo.

Métricas mensuradas empregando Validação Cruzada por Série Temporal (*TimeSeries Split*, k=5):

| Modelo | Escopo de Features | Acurácia (TSCV) | F1-Score (Macro) |
|--------|---------------------|-----------------|------------------|
| **M1** | Apenas Pré-Jogo (Histórico Médio) | 48.98% | 40.05% |
| **M2** | Apenas Intra-jogo (1º Tempo)      | 63.02% | 53.79% |
| **M3** | Combinado (Histórico + 1º Tempo)  | 63.59% | 54.41% |

*Diagnóstico de Evolução:* 
1. A transição de M1 para M2 demonstra que métricas extraídas puramente do 1º tempo da partida são atributos de previsão estritamente superiores ao retrovisor das últimas partidas (uma evolução de 48.98% para 63.02% em acurácia). 
2. A integração efetuada em M3 comprova redundância nos dados e estabilidade da acurácia com uma leve otimização no ganho analítico do modelo estocástico de validação.

## 4. Análise Sistêmica e Fatores de Erro

Observou-se através dos *logs* de análise de erro que a maior parcela de divergência estrita (confiança acima de 90% em uma classe com registro real em classe dissimulada) originava-se de assimetrias estatísticas no pré-jogo que não se traduziram nos escores, exemplificando a variância extrema inerente à conversão de oportunidades (*xG*) e cenários episódicos de um esporte de pontuação limitante e estocástica.

O fluxo preditivo multiclasse validado substitui todas as formulações com metodologias de avaliação e alvos binários das iterações originais do projeto. A fundamentação comprova o avanço metodológico de processamento analítico para a construção formal descrita em monografia acadêmica.
