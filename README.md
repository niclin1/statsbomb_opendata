# Predição de resultado de partidas de futebol no intervalo

Repositório do Trabalho de Conclusão de Curso em Engenharia de Computação, CEFET-MG, campus
Timóteo. O trabalho prevê o resultado de uma partida de futebol no intervalo, a partir de métricas
táticas do primeiro tempo extraídas dos dados abertos de eventos da StatsBomb.

Orientador: Prof. Dr. Marcelo de Sousa Balbino.

## O problema

O modelo recebe apenas o que aconteceu nos 45 minutos iniciais, sem o placar, e classifica o
resultado final da partida em três classes, do ponto de vista de uma das equipes: Derrota, Empate
ou Vitória. O momento da predição é o intervalo, quando uma intervenção tática ainda é possível.

## Atributos

O vetor tem 11 atributos por equipe-partida. Três métricas táticas calculadas em cada um dos três
blocos de 15 minutos do primeiro tempo, mais duas contagens de cartão vermelho:

| métrica | blocos | zona de cálculo |
|---|---|---|
| xG | 0–15, 15–30, 30–45 | soma de `shot_statsbomb_xg` das finalizações da equipe |
| PPDA | 0–15, 15–30, 30–45 | passes do adversário em `x ≤ 60` sobre ações defensivas da equipe em `x > 60` |
| Field Tilt | 0–15, 15–30, 30–45 | passes no terço final, `x ≥ 80`, sobre o total das duas equipes |
| cartão vermelho | primeiro tempo inteiro | duas contagens: própria equipe e adversário |

O modelo não recebe o xG nem o PPDA do adversário. Um diagnóstico mediu o efeito de acrescentá-los:
2,25 pontos percentuais de F1 macro em validação cruzada, favorável em 4 de 5 dobras. O conjunto de
teste já foi aberto, então a mudança está registrada como trabalho futuro e não foi incorporada ao
modelo final.

## Base de dados

2.635 partidas, entre agosto de 2010 e julho de 2024, em nove competições. A base é espelhada: cada
partida gera duas linhas, uma por equipe, com alvos complementares, o que dá 5.270 linhas e **n
efetivo de 2.635**. Toda partição e toda validação cruzada agrupa por `match_id`.

O domínio primário é a temporada 2015/16 das cinco grandes ligas europeias, 1.823 partidas, com
corte temporal por data:

| partição | partidas | janela |
|---|---|---|
| treino | 1.196 | 2015-08-07 a 2016-02-14 |
| validação | 287 | 2016-02-17 a 2016-04-02 |
| teste | 340 | 2016-04-03 a 2016-05-17 |

As 812 partidas restantes ficam em quatro conjuntos de validação externa, cada um isolando um
deslocamento distinto: elite pós-2016 (264), La Liga 2010–2015 (171), Indian Super League (115) e
seleções (262). Nenhum deles participa do treino nem da seleção de hiperparâmetros.

## Protocolo de avaliação

O conjunto de teste foi avaliado **uma única vez**, após a seleção de hiperparâmetros sobre a
partição de validação. Os intervalos de confiança de 95% vêm de reamostragem *bootstrap* de
partidas, que preserva a dependência entre as duas linhas de cada partida.

Cada execução grava um `manifest.json` em `outputs/runs/<run_id>/` com o SHA-256 do conjunto de
dados, o commit do git, as versões das bibliotecas e a semente. As tabelas em `outputs/tabelas/`
são geradas por código e não devem ser editadas à mão.

### Sobre vazamento de dados

A versão anterior deste trabalho (TCC I) **tinha** vazamento na seleção de hiperparâmetros: a busca
com `TimeSeriesSplit` percorria a base inteira, incluindo o período usado como teste, e a estimativa
de cerca de 56% de acurácia resultante era otimista. O particionamento em três subconjuntos e a
regra de avaliação única existem para corrigir esse ponto, e a diferença entre as duas estimativas é
um dos resultados do trabalho. A auditoria está registrada na execução
`20260811T025712_wp0_auditoria` (ver "Execuções e procedência", abaixo) e descrita na monografia.

## Estrutura

```text
statsbomb_opendata/
├── src/
│   ├── common/            config.py (configuração e UTF-8), run.py (manifest de execução)
│   ├── etl/momentum_v5.py ingestão da StatsBomb e cálculo das métricas por bloco
│   ├── modeling/
│   │   ├── splits.py      partições por data, com agrupamento por partida
│   │   ├── baselines.py   acaso uniforme, classe majoritária e mando de campo, por partição
│   │   ├── train.py       otimização de hiperparâmetros (Optuna)
│   │   ├── evaluate.py    avaliação final, com guarda de abertura única do teste
│   │   ├── explain.py     valores SHAP das três classes
│   │   └── wp5_*, wp6_*   segundo tempo e calibração
│   └── reporting/         audit_dataset.py e tables.py (geração das tabelas .tex)
├── tests/                 suíte de regressão (pytest)
├── outputs/
│   ├── runs/<run_id>/     manifest.json e resultados de cada execução
│   ├── tabelas/           tabelas .tex geradas por código
│   ├── figuras/
│   ├── consultas/         consultas à base aberta (catálogo, eventos por partida)
│   └── teste_aberto.json  guarda da abertura única do conjunto de teste
├── data/momentum_data.csv base de atributos usada em todas as execuções
├── docs/                  relatórios das iterações do TCC I
└── config/experiment.yaml decisões de experimento versionadas
```

O texto da monografia, as notas de trabalho e o material de referência não fazem parte deste
repositório.

## Execuções e procedência

Cada execução de experimento grava uma pasta `outputs/runs/<run_id>/`. O identificador é a data e a
hora de início em UTC, no formato `AAAAMMDDTHHMMSS`, seguidas do nome do experimento. Por exemplo,
`20260903T030806_wp4_shap` é a análise SHAP iniciada em 3 de setembro de 2026 às 03:08:06 UTC. O
`manifest.json` de cada pasta registra a semente aleatória, as versões das bibliotecas, o SHA-256 da
base (`data/momentum_data.csv`), a configuração usada e o commit do repositório no momento da
execução. Os CSVs da pasta trazem os resultados que alimentam as tabelas de `outputs/tabelas/`.

Os manifests registram o commit `f8ed5b7` com a marca `"sujo": true`: o código das execuções tinha
alterações ainda não versionadas naquele momento. Esse código foi publicado depois na marcação
`tcc2-defesa`, que é a versão citada na monografia. Depois da última execução (4 de setembro de
2026) mudaram só dois arquivos: os rótulos de texto das tabelas em
`src/modeling/efeito_composicao.py` ("TCC I" passou a "etapa anterior") e a etapa tipográfica
`src/reporting/formatar_tabelas_tex.py`, que formata as tabelas para a monografia e não recalcula
nenhum número.

| execução | experimento |
|---|---|
| `20260811T025712_wp0_auditoria` | auditoria da base |
| `20260818T223606_wp2_otimizacao` | otimização de hiperparâmetros (Optuna) |
| `20260903T030017_wp3_avaliacao_teste` | avaliação única no conjunto de teste |
| `20260903T025819_wp3_avaliacao_teste` | tentativa interrompida antes da avaliação (só a matriz de confusão) |
| `20260903T030806_wp4_shap` | valores SHAP por classe e teste de simetria do empate |
| `20260903T034155_wp5_segundo_tempo` | extensão dos atributos ao segundo tempo |
| `20260903T034837_wp6_calibracao` | calibração (ECE e Brier) |
| `20260904T023603_diag_atributos_adversario` | diagnóstico com atributos do adversário |
| `20260904T024438_efeito_composicao` | efeito da composição do conjunto de teste |

A base derivada `data/momentum_data.csv` foi calculada a partir dos [dados abertos da
StatsBomb](https://github.com/statsbomb/open-data), cujos termos de uso pedem a atribuição da fonte.

Os arquivos soltos em `src/` (`etl_pipeline.py`, `features_*.py`, `train_xgboost.py`,
`train_optuna.py`, entre outros) são as iterações 1 a 5 do TCC I. Ficam preservados como registro
histórico, são citados no capítulo de metodologia e não fazem parte do pipeline atual.

## Reprodução

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q
```

Auditoria da base, sem tocar em modelo:

```bash
python -m src.reporting.audit_dataset
```

Partições e baselines:

```bash
python -m src.modeling.splits
```

```bash
python -m src.modeling.baselines
```

Otimização de hiperparâmetros:

```bash
python -m src.modeling.train --n-trials 50
```

Explicabilidade das três classes:

```bash
python -m src.modeling.explain --particao teste
```

O comando de avaliação final (`src.modeling.evaluate`) tem guarda de abertura única: reexecutar
sobre a partição de teste exige a opção `--reabrir` com uma justificativa, que fica registrada no
manifest. O teste já foi aberto e os modelos não podem ser alterados.

## Métricas descartadas

Convex Hull, Zone 14 Passes e Deep Completions foram avaliadas nas iterações iniciais e não entram
no modelo final. As duas primeiras não discriminaram o resultado; a terceira foi descartada após a
terceira iteração. O registro das alternativas testadas, incluindo os resultados nulos, está nos
relatórios de `docs/` e na monografia.
