# Planejamento Técnico e Roadmap Estratégico para a Iteração 04: Rigor Científico, Expansão de Dados e Otimização Isolada

## 1. O Problema Metodológico (Controle de Variáveis)
Na transição para a Fase 04 do projeto, deparamo-nos com um desafio clássico do método científico aplicado ao *Machine Learning*: **o risco de sobreposição de variáveis**. 

Se executarmos a expansão do dataset, a mudança para predição Multiclasse (Vitória/Empate/Derrota), a adição de fragmentos de 15 minutos e a otimização com Optuna *simultaneamente*, perderemos a capacidade de rastrear a causalidade do ganho (ou perda) de desempenho. Se a acurácia subir 10%, foi por causa dos dados novos? Das fatias de 15 minutos? Ou pelo Optuna? 

Para a monografia (e para alimentar o NotebookLM com lógica irrepreensível), precisamos de um **Estudo de Ablação** (*Ablation Study*). Vamos adicionar uma complexidade por vez e registrar o delta de desempenho de cada uma.

## 2. Roadmap Científico da Iteração 04

A Iteração 04 será dividida em 4 subfases restritas:

### Fase 4.1: Expansão Universal (O Grande Rerun)
*   **Ação:** Executar os pipelines atuais (Baseline Pré-jogo e In-Play 1º Tempo) sobre toda a base de dados disponível da StatsBomb.
*   **Por que isolar:** Precisamos estabelecer a métrica base do modelo In-Play com $N$ máximo antes de adicionar qualquer nova métrica tática. Isso provará se o modelo atual generaliza bem em larga escala.
*   **Entrega:** Registro das métricas (Accuracy, ROC-AUC) do modelo In-Play base rodando no dataset completo.

### Fase 4.2: Reformulação do Problema (Binary vs. Multiclass)
*   **Ação:** Alterar o target do modelo da Fase 4.1 para cenário multiclasse (Vitória, Empate, Derrota).
*   **Por que isolar:** O empate é uma característica fundamental do futebol. Mudar a função objetivo do XGBoost (`multi:softprob`) impactará as métricas drasticamente (a acurácia natural cai em problemas de 3 classes). Precisamos documentar o "tombo" de precisão ao tornar o problema mais realista.
*   **Entrega:** Registro do desempenho Multiclasse com as métricas do 1º tempo inteiro.

### Fase 4.3: Engenharia de Micro-Momentos (Resolução Temporal)
*   **Ação:** Segmentar as métricas do 1º tempo em janelas de 15 minutos (0-15', 15-30', 30-45') para capturar o momento (tendência de crescimento do *Field Tilt* e *xG*) e adicionar Cartões Vermelhos no 1º tempo.
*   **Por que isolar:** Aqui respondemos diretamente a um dos Objetivos Específicos da tese ("resoluções temporais progressivas"). Comparamos este resultado com a Fase 4.2 para provar matematicamente para a banca que fatiar o jogo captura o **Momento Tático** e melhora a previsão.
*   **Entrega:** Avaliação do ganho preditivo dos micro-momentos.

### Fase 4.4: Otimização de Hiperparâmetros (Optuna)
*   **Ação:** Aplicar Otimização Bayesiana através do Optuna no modelo exato construído na Fase 4.3.
*   **Por que isolar:** Somente otimizando hiperparâmetros no modelo final validado poderemos afirmar na monografia o impacto **exato** e isolado do Optuna. Provaremos o quanto os hiperparâmetros padrão (*default*) do XGBoost estavam "segurando" o potencial do modelo. 
*   **Entrega:** Acurácia da Fase 4.3 vs. Acurácia da Fase 4.4. Extração do gráfico de *Feature Importance* (SHAP) finais.

## 3. Conclusão Metodológica
Esta separação metodológica garante que cada parágrafo da sua Discussão de Resultados na Monografia tenha um respaldo empírico isolado. O orientador e a banca verão a evolução cronológica: Os dados deram robustez; O multiclasse deu realismo; Os 15 minutos deram o contexto dinâmico; E o Optuna extraiu o ápice matemático do pipeline.
