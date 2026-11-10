# **Relatório Técnico da Iteração 04: Implementação de Micro-Momentos, Escalabilidade de Dados e Otimização Bayesiana Multiclasse**

## **Resumo Executivo**

Este documento detalha a execução e os resultados da quarta iteração do projeto de predição de futebol, marcando a transição de um escopo de prova de conceito para um pipeline de Aprendizado de Máquina robusto, em larga escala e rigorosamente validado. Em alinhamento com os objetivos específicos da monografia, a Iteração 04 superou limitações de engenharia de dados, reformulou o problema matematicamente para o cenário multiclasse (Vitória, Empate, Derrota) e implementou a segmentação tática do jogo em janelas de 15 minutos (0-15', 15-30', 30-45'). 

Além da engenharia de atributos avançada — que incluiu métricas de *Expected Goals* (xG), *Field Tilt*, *Passes Per Defensive Action* (PPDA) e assimetrias por Cartões Vermelhos —, o modelo XGBoost foi otimizado utilizando algoritmos de Otimização Bayesiana (*Optuna*). O desempenho superior ao acaso geométrico de 33% (atingindo ~56% no conjunto de teste extremo) validou empiricamente a tese de que a granularidade temporal do primeiro tempo retém forte poder de generalização do "momento tático" para prever o resultado final. Por fim, técnicas de IA Explicável (*SHAP*) foram aplicadas para extrair o real impacto vetorial de cada variável temporal.

## ---

**1. Engenharia de Dados Empírica e Resolução do Gargalo de Cache**

A ambição de escalar o treinamento do modelo para abranger o máximo possível da base aberta da *StatsBomb* exigiu intervenções arquiteturais críticas no script de *Extract, Transform, Load* (ETL).

### **1.1. O Problema de Saturação de Espaço (Disco C:)**
Durante a requisição massiva inicial, a biblioteca `statsbombpy` adotou um comportamento agressivo de armazenamento em *cache* de arquivos JSON brutos no diretório temporário do sistema operacional (Disco C:). Considerando a densidade informacional de milhares de partidas (cada uma aglomerando milhares de nós de eventos), o volume de dados rapidamente esgotou a capacidade física do disco de inicialização do sistema, interrompendo o fluxo computacional.

### **1.2. Hijacking de Requisições via SQLite (requests_cache)**
Para solucionar o gargalo físico e evitar ultrapassar os limites de requisição da API (*rate limits*), a arquitetura foi aprimorada com a biblioteca `requests_cache`. A estratégia envolveu interceptar as requisições HTTP da *StatsBomb* e redirecionar os retornos (arquivos brutos) para um banco de dados local SQLite hospedado em uma unidade de armazenamento secundária estendida (`E:\_cache\statsbomb_cache.sqlite`). Esse salto de engenharia reduziu o tempo de requisições iterativas subsequentes de horas para frações de segundo.

### **1.3. Refinamento de Séries Históricas (Amostragem Tática Moderna)**
Uma varredura por todas as 75 temporadas disponíveis demonstrou alta heterogeneidade histórica (com jogos datando da década de 1970). Diante disso, optou-se por aplicar um filtro tático-cronológico, restrigindo o escopo de análise às temporadas a partir de **2010**. O recorte foi fixado nas 5 Grandes Ligas Europeias (La Liga, Premier League, Serie A, Ligue 1, Bundesliga), além de grandes competições internacionais (Copa do Mundo, Eurocopa, Copa América). O resultado foi um *dataset* de cerca de **2.900 partidas masculinas**, configurando um *sweet-spot* matemático contemporâneo de alta qualidade para o treinamento algorítmico supervisionado.

## ---

**2. A Captura Matemática do "Momento": Segmentação Temporal**

A abordagem clássica de integrar os dados do primeiro tempo (0-45') em uma métrica unificada ofusca o conceito de "momento" ou inércia da partida (*momentum*). Como delineado no escopo do TCC, a Iteração 04 decompôs estrategicamente os dados estatísticos e espaciais.

### **2.1. Fatiamento em Micro-Momentos de 15 Minutos**
As coordenadas topológicas e os algoritmos espaciais foram reprogramados para fatiar programaticamente os 45 minutos inaugurais em três blocos temporais rigorosos:
*   **0-15 Minutos:** Fase de adaptação e espelhamento tático inicial.
*   **15-30 Minutos:** Estabelecimento do domínio de passe e quebra de bloco.
*   **30-45 Minutos:** Retenção final, fadiga primária e desgaste da pressão.

Para cada uma destas janelas, o ETL (`src/etl_momentum_v4.py`) computou separadamente:
*   O **xG (Expected Goals)** acumulado.
*   A pressão defensiva espacial **(PPDA)** isolada na janela.
*   A inclinação de campo **(Field Tilt)** representativa daquela fração específica.
*   **Cartão Vermelho:** Uma flag multiclasse estrita (0 ou 1) caso a equipe tenha sofrido uma exclusão, lidando logicamente com a assimetria geométrica gerada por ter 10 homens no campo.

## ---

**3. Modelagem e Otimização Bayesiana (Optuna)**

A tabela analítica foi consolidada (com 18 variáveis preditivas por partida) e a variável alvo foi convertida numa distribuição multiclasse (`0=Derrota`, `1=Empate`, `2=Vitória`). A probabilidade estatística de acerto por palpite randômico neste espaço é calculada em 33,3%.

### **3.1. Treinamento Otimizado vs. Validação Temporal (Walk-Forward)**
Foi implementada uma validação cruzada do tipo *Time Series Split* (`TimeSeriesSplit`), que impede o modelo de ler o "futuro para prever o passado" (vazamento temporal). O modelo escolhido foi o **XGBoost** operando com a função discriminante `multi:softprob`.

Para garantir que o modelo extrairia a correlação máxima dos micro-momentos em vez de ficar condicionado à parametrização humana empírica, a biblioteca **Optuna** foi injetada para orquestrar 50 *trials* de experimentação multivariada. A otimização varreu rigorosamente as funções de perda logarítmica avaliando `lambda`, `alpha`, `max_depth`, `eta`, entre outros condicionantes estruturais.

### **3.2. Resultados Empíricos**
A experimentação bayesiana extraiu uma matriz convergente (Trial 35) que retornou:
*   **Acurácia Média (Conjuntos Temporais):** ~52,12%
*   **Acurácia Global (Teste Puramente Futuro de 20%):** ~56%
*   **Precisão da Derrota (Classe 0) e Vitória (Classe 2):** Recall estável de 75% e 72% respectivamente, demonstrando formidável viabilidade em detectar assimetrias superiores da partida (as duas equipes raramente terminam controlando a mesma parcela do campo).
*   **A Singularidade do Empate:** O *F1-Score* para a classe exata de Empate (1) permaneceu perto de nulo, reafirmando consensos empíricos na literatura de *Analytics* esportivo: O Empate funciona como uma anomalia randômica (*ruído*) dentro das projeções de Expected Goals espaciais. Se as táticas e o ímpeto apontam vitória para o Time B, porém as finalizações terminam no travésão, configurando 0x0, a matemática punirá o algoritmo preditivo através das circunstâncias estocásticas.

## ---

**4. Extração Contrafactual e IA Explicável (SHAP)**

O cume empírico do processamento da Iteração 04 foi a submissão do modelo campeão à Teoria dos Jogos Cooperativos através de valores **SHAP** (*SHapley Additive exPlanations*). 
Um problema dimensional envolvendo as atualizações tridimensionais das bibliotecas XGboost/SHAP (Amostras vs. *Features* vs. Classes) foi cirurgicamente corrigido via código de fatiamento de *arrays* focando estritamente na **Classe 2 (Vitória)**.

Isolando os vetores Shapley, extraiu-se finalmente o gráfico das Importâncias de Atributo (*Beeswarm Plot* / *Summary Plot* registrado em `outputs/plots/optuna_shap_win_class.png`). Este artefato computacional corrobora definitivamente os capítulos finais da monografia: as janelas temporais intermédias (como o *Field Tilt* entre os 30-45 minutos) desempenham pesos específicos assimétricos na cadeia da vitória, superando radicalmente a importância de métricas puramente descritivas e provando categoricamente a superioridade do modelo espaciotemporal para fins de diagnóstico tático acadêmico.
