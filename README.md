# Projeto StatsBomb


Bem-vindo ao repositório do **Projeto StatsBomb**, desenvolvido como Trabalho de Conclusão de Curso (TCC). O objetivo deste projeto é construir um sólido pipeline de Extração de Dados (ETL), Inteligência Visual Tática e Modelagem Preditiva (*Machine Learning*) focado em prever resultados de partidas de futebol profissional utilizando dados de eventos abertos da **StatsBomb**.

## Sobre o Projeto

A premissa fundamental do TCC propõe que estatísticas arcaicas de futebol (ex: "posse de bola bruta" ou "total de chutes") falham em capturar a natureza complexa e estocástica do esporte. 

O projeto inova ao ingerir dados espaciais detalhados e transformá-los em métricas avançadas de eficácia e domínio geométrico, tais como:
- **Field Tilt** (Inclinação de Campo)
- **PPDA** (Passes Per Defensive Action / Intensidade de Pressão)
- **Deep Completions** (Passes Efetivos Próximos ao Gol)
- **Expected Goals (xG)**
- **Zone 14 & Half-Spaces** (Mapeamento de Heatmaps)
- **Convex Hulls** (Evolução Territorial)

Através da modelagem em **Séries Temporais (Sinalizações de Forma Recente em Média Móvel de 5 jogos)** e validação temporal *Walk-Forward*, superamos o desafio metodológico clássico de *Data Leakage*, elevando a precisão da inteligência artificial.

## Estrutura do Repositório

```text
statsbomb_opendata/
│
├── docs/               # Documentação acadêmica, laudos estatísticos e Roadmap detalhado
│
├── src/                # Código-fonte principal do projeto
│   ├── data_loader.py            # Validação inicial da API Statsbomb
│   ├── etl_pipeline.py           # Algoritmo de ingestão em massa (Team-Match level)
│   ├── features_advanced.py      # Cálculos geométricos pioneiros (Convex Hull, Field Tilt)
│   ├── features_spatial.py       # Funções para segmentação do campo (Zonas de Pitch)
│   ├── features_time_series.py   # Transformação dos dados estatísticos via Janela Deslizante (Rolling Averages)
│   ├── train_baseline.py         # Iteração 01: Teste inicial do LogReg provando a "Patologia da Amostra (N=32)"
│   ├── train_xgboost.py          # Iteração 02: Treinamento final com Random Forest e XGBoost
│   └── visualize_advanced.py     # Subrotina mplsoccer (Visualização e Mapas de Calor do TCC)
│
└── requirements.txt    # Dependências de ambiente Python
```

## Como Reproduzir (Para Avaliadores e Orientadores)

Siga os passos abaixo para construir a pipeline inteira localmente. 

**1. Instalação de Dependências**
```bash
pip install -r requirements.txt
```

**2. Fase 1: Extração de Dados (ETL)**
Para baixar as competições da API da StatsBomb e criar as agregações `Team-Match` contendo Field Tilt, xG e PPDA calculados nativamente:
```bash
python src/etl_pipeline.py
```
*(Nota: Gera `historical_match_data.csv` na pasta `data/`)*

**3. Fase 2: Inteligência Visual e Mapeamento**
Para desenhar mapas de densidade comparativos e geometria dos times (estudo de caso: Argentina x França, Final da Copa 2022):
```bash
python src/visualize_advanced.py
```
*(Nota: As imagens ficarão sob a pasta `outputs/plots/`)*

**4. Fase 3: Engenharia de Features Temporais**
Para alinhar cronologicamente todos os eventos, evitar *Data Leakage* e preparar os vetores focados no Histórico Recente (janela de 5 partidas) do time e seus adversários diretos:
```bash
python src/features_time_series.py
```
*(Nota: Gera `modeling_data.csv`, eliminando os primeiros jogos nulos das equipes)*

**5. Fase 4: Treinamento de Machine Learning e Explainability**
Para testar **Random Forest** vs **XGBoost** avaliando temporalmente o resultado das métricas táticas e descobrir quais Features pesaram mais nas vitórias:
```bash
python src/train_xgboost.py
```

## Principais Descobertas e Marcos (Iteração 02)

O projeto demonstrou matematicamente a invalidade de treinar previsões agregadas no nível de jogador em eventos singulares (**Iteração 01 obteve 0% de acurácia**). 

Ao corrigir a variável independente para o retrospecto da equipe inteira (**Iteração 02**), aliando as poderosas métricas como **Pressão (PPDA)** e **Gols Esperados (xG)**, o algoritmo **XGBoost Classifier estabeleceu 65.00% de Acurácia genuína**, demonstrando competência notável em prever eventos essencialmente imprevisíveis como o esporte interativo dinâmico.

Para aprofundamento científico total destas passagens e limitações testadas, consulte a pasta `/docs`.
