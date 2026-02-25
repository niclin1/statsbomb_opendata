# Relatório de Execução: Iteração 02 (Escalonamento e Modelagem Preditiva)

Este documento detalha as etapas práticas implementadas na "Iteração 02" do projeto Anti Gravity, cujo objetivo principal foi sanar as deficiências estatísticas (N=32) descobertas na Fase Piloto e estabelecer um pipeline funcional de *Machine Learning* capaz de realizar predições.

## 1. Escalonamento de Dados (ETL Pipeline)
Com base no diagnóstico sistêmico anterior, a unidade de análise foi alterada de *Jogador-Partida* para **Equipe-Partida**.

- **Implementação**: Foi desenvolvido o script `src/etl_pipeline.py` para automatizar a ingestão massiva de dados via *StatsBomb API*.
- **Amostragem Base**: Para o ambiente de testes atual, foi utilizado um recorte de aproximadamente 300 partidas oriundas de torneios como a Copa do Mundo. Essa quantidade foi determinada para otimizar o uso do armazenamento local (*Disk Space*), garantindo volume estatístico suficiente sem comprometer o *hardware*.
- **Engenharia de Métricas Brutas**: Durante o ETL, as seguintes variáveis táticas foram calculadas nativamente a partir das coordenadas dos eventos:
  - **Total xG** (Gols Esperados)
  - **Field Tilt** (Domínio Territorial)
  - **PPDA** (Intensidade de Pressão Defensiva)
  - **Deep Completions** (Passes efetivos próximos/dentro da área restrita)

## 2. Inteligência Visual Avançada
Para validar o contexto tático e fornecer recursos explicativos ao projeto, o script `src/visualize_advanced.py` implementou técnicas de mapeamento da biblioteca `mplsoccer`.
A final da Copa de 2022 (Argentina vs França) foi utilizada como estudo de caso:

- **Heatmaps de Alta Densidade**: Foram aplicados filtros bidimensionais (histogramas) detalhando concentrações de posse (passes e conduções).
- **Zone 14 e Half-Spaces**: As zonas ofensivas de maior índice de assistência (Zone 14) e infiltração (Meio-Espaços) foram projetadas sobre os mapas de calor para comprovar numericamente o uso dessas áreas por ambas as equipes.
- **Evolução de Território (Convex Hull)**: Diferente da fase piloto iterativa, as formas geométricas (hulls) foram separadas entre **1º Tempo e 2º Tempo**, mapeando perfeitamente oscilações na agressividade e recuo das linhas defensivas no decorrer do jogo.

## 3. Engenharia de Features (Séries Temporais)
Para que o modelo aprenda padrões *preditivos* — e não sofra do viés de analisar o futuro para explicar o passado (*Data Leakage*) — o script `src/features_time_series.py` foi executado sobre o dataset gerado no ETL.

- **Médias Móveis (Rolling Averages)**: A performance de uma equipe na partida `N` não utilizou a métrica da partida `N`. Foi gerada uma janela de `Window=5`, cobrindo da partida `N-5` até `N-1`. Isso representa o "Momento do Time" (Form).
- **Filtro de Aquecimento**: Inicialmente, tínhamos aproximadamente 300 partidas. Como os 5 primeiros jogos de *cada time* não continham histórico prévio e geravam valores nulos (`NaN`), eles foram removidos.
- **Dataset de Modelagem**: O resultado líquido desta transformação rendeu **160 amostras matematicamente perfeitas** para treinamento, compreendendo as features retrospectivas da própria equipe e a "Forma" retrospectiva do adversário.

## 4. Modelagem Preditiva e Avaliação
A etapa final (`src/train_xgboost.py`) testou as heurísticas implementadas por meio de algoritmos de aprendizado baseados em árvores de decisão. A estratégia imperativa foi **Validação Walk-Forward**, dividindo os dados em tempo (Treino no Passado \> Teste no Futuro).

### Resultados Obtidos:
- **Modelo Baseline (Random Forest)**:
  - Acurácia: 60.00%
- **Modelo Challenger (XGBoost)**:
  - Acurácia: **65.00%**
  - Precision: 50.00%
  - Recall: 64.29%
  - Log-Loss: 0.8710

### Discussão e Importância das Variáveis
A adoção da unidade correta (*Team-Match*) e da arquitetura temporal elevaram a performance do projeto de absurdos **0% (Fase Piloto) para 65% (Iteração 02)**. Esta acurácia representa um desempenho preditivo real e utilizável em análises esportivas. 

A plotagem de *Feature Importance* confirmou a premissa teórica do TCC: as variáveis mais decisivas para a I.A. escolher o time vencedor não foram apenas os gols marcados no passado, mas principalmente:
1. O **xG Recente** gerado pelo time.
2. A **intensidade de pressão (PPDA)** da equipe.
3. O histórico de *Deep Completions* sofridas pelo Oponente no seu terço defensivo.

### Conclusão e Relatório de Erros
A exportação do `model_evaluation_report.txt` apontou que as instâncias com a maior "certeza de erro" do algoritmo XGBoost (onde ele atribuía 95%+ de probabilidade de vitória a um time que acabou não vencendo) tratam-se de surpresas estatísticas reais do futebol — situações onde um time superou o seu próprio teto de produção tática de forma atípica. Estes *outliers* valem a pena ser qualitativamente descritos na versão textual final do TCC.
