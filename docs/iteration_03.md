# **Planejamento Técnico e Roadmap Estratégico para o projeto: Iteração 03 — Predição In-Play e Análise de Estado de Jogo**

## **1. Diagnóstico e Motivação da Iteração 03**

A Iteração 02 demonstrou com sucesso que a modelagem preditiva de futebol baseada em dados histórico-retrospectivos é estatisticamente viável, atingindo **65% de acurácia** com o algoritmo XGBoost e um conjunto de variáveis de forma recente (*rolling averages* de xG, PPDA, Field Tilt). No entanto, o modelo resultante opera no paradigma **Pré-Jogo** (*Pre-Match Prediction*): todas as *features* utilizadas descrevem o comportamento *passado* das equipes em partidas *anteriores*, sendo utilizadas para antever o resultado de uma partida *ainda não disputada*.

A pergunta levantada pelo orientador do projeto expõe uma limitação conceitual relevante e potencialmente produtiva para a tese:

> *"Seria possível e melhor prever partes de uma partida durante o jogo e então obter o resultado final?"*

Esta questão introduz o paradigma da **Predição In-Play** (*In-Match Prediction*), vastamente explorado pela indústria de análise esportiva e amplamente utilizado em plataformas de apostas em tempo real, sistemas de *coaching* dinâmico e cobertura jornalística assistida por IA. 

A hipótese central desta iteração é a seguinte:

> *Os dados táticos gerados durante o 1º Tempo de uma partida (como xG acumulado, PPDA no bloco defensivo e inclinação territorial de campo) carregam um sinal preditivo significativo sobre o resultado final, que complementa e potencialmente supera o sinal capturado pelas features de histórico pré-jogo.*

---

## **2. Fundamentação Teórica: O Paradigma In-Play na Análise Esportiva**

### **2.1. Limitações Intrínsecas da Predição Pré-Jogo**

O modelo baseado exclusivamente em histórico (Iteração 02) é estruturalmente "cego" ao estado atual do jogo. Ele assume que duas equipes que se enfrentam com determinado histórico reproduzirão, em média, esses padrões históricos — desconsiderando completamente o que acontece dentro das quatro linhas. Esta cegueira tem um custo real: situações de reviravolta (*comebacks*), impacto de expulsões, mudanças táticas no intervalo e fadiga física são fenômenos que o histórico pré-jogo não consegue capturar.

A literatura em ciência do esporte (*Sports Science*) identifica que uma parcela significativa da variância no resultado de uma partida ocorre durante o próprio jogo, sob a influência de eventos contingentes (gols sofridos, cartões, substituições táticas). O modelo de Iteração 02 descarta toda essa variância por construção.

### **2.2. Segmentação Temporal da Partida**

A base de dados da StatsBomb fornece, para cada evento registrado, a variável `period`, que mapeia os eventos cronologicamente:

- `period == 1`: Primeiro Tempo
- `period == 2`: Segundo Tempo
- `period == 3` e `period == 4`: Prorrogação (quando aplicável)
- `period == 5`: Pênaltis

Esta granularidade temporal permite uma segmentação precisa da partida em janelas discretas de análise, sem ambiguidade e sem necessidade de aproximações temporais baseadas em `timestamp`.

### **2.3. A Arquitetura de Estado de Jogo (*Game-State*)**

O conceito de **Game-State** é a pedra angular da modelagem In-Play. O estado do jogo em qualquer momento `t` pode ser formalizado como um vetor de variáveis que descrevem a situação corrente:

- **Placar Parcial** (`score_diff`): Diferença entre gols do time e do adversário até o momento `t`.
- **xG Acumulado** (`cumulative_xg`): Qualidade de oportunidades criadas/cedidas até `t`.
- **Pressão Aplicada** (`ppda_period`): Intensidade de pressão defensiva no período ativo.
- **Domínio Territorial** (`field_tilt_period`): Inclinação de campo no período ativo.

A hipótese é que o vetor de Game-State no intervalo (ao final do 1º Tempo) é um preditor forte do resultado final, pois captura tanto a qualidade do desempenho (xG, Field Tilt) quanto o contexto situacional (placar parcial).

---

## **3. Arquitetura de Dados: Unidade de Análise Intra-Jogo**

### **3.1. Refatoração do ETL (Unidade: Equipe-Partida-Período)**

A principal diferença técnica entre a Iteração 02 e a Iteração 03 está na **chave de agregação** do Pipeline ETL. Enquanto na Iteração 02 agrupávamos por `(team_name, match_id)`, na Iteração 03 agrupamos por:

```
(team_name, match_id, period)
```

Isso produz, para cada jogo, **4 linhas ao invés de 2** (2 times × 2 tempos), tornando cada linha uma observação do desempenho de uma equipe dentro de um único período.

### **3.2. Métricas Calculadas por Período**

O script `src/etl_intra_match.py` calculará as seguintes variáveis para cada bloco (equipe, partida, período):

| Feature | Descrição Técnica |
| :------------------ | :--------------------------------------------------------- |
| `period` | Período da partida (1 = 1º Tempo, 2 = 2º Tempo) |
| `field_tilt_period` | Field Tilt calculado apenas nos eventos do período |
| `xg_period` | Soma de `shot_statsbomb_xg` nos chutes do período |
| `ppda_period` | Passes por Ação Defensiva restrito ao período |
| `shots_on_target` | Total de chutes no alvo no período |
| `deep_completions` | Passes finalizados dentro da área restrita no período |
| `score_diff_period` | Diferença de gols marcados no período (contexto situacional)|

### **3.3. Engenharia de Features de Momentum**

Um componente central da Iteração 03 é a criação de **features de momentum** — variáveis que capturam a variação de desempenho entre os dois tempos. Estas serão calculadas no script `src/features_inplay.py`:

- **`delta_xg`**: `xg_period_2 - xg_period_1` (time aumentou ou perdeu criatividade ofensiva?)
- **`delta_field_tilt`**: Variação do domínio territorial entre os tempos
- **`delta_ppda`**: Variação da intensidade de pressão (o time "fechou" mais no 2º Tempo?)

Estas features de delta são incomuns na literatura padrão pré-jogo, pois exigem a conclusão do 1º Tempo antes de serem calculadas, confirmando o caráter intra-jogo do modelo.

---

## **4. Estratégia de Modelagem Comparativa**

O diferencial científico desta iteração é o **desenho experimental comparativo**. Ao invés de simplesmente treinar um novo modelo, estruturaremos três cenários mutuamente exclusivos que serão avaliados sob as mesmas condições de split temporal:

| Modelo | Features | Paradigma |
| :----- | :------- | :-------- |
| **M1 — Pré-Jogo (Baseline)** | Rolling averages históricos (Iteração 02) | Pré-Jogo |
| **M2 — 1º Tempo** | xG, PPDA, Field Tilt do 1º Tempo + placar parcial | In-Play (Intervalo) |
| **M3 — Combinado** | M1 + M2 (histórico + estado no intervalo) | Pré-Jogo + In-Play |

Esta estrutura permite responder precisamente à hipótese do orientador:

- Se **M2 > M1**: Os dados intra-jogo superam o histórico pré-jogo. A predição *durante* a partida é intrinsecamente mais eficaz.
- Se **M3 > M2 > M1**: Ambos os sinais são complementares; o modelo combinado é o mais robusto.
- Se **M1 ≈ M2**: O histórico e o estado de jogo carregam sinais redundantes, o que sugere que os padrões táticos de 5 jogos já "preveem" como o time atuará no 1º Tempo.

---

## **5. Scripts Planejados**

| Script | Função |
| :---------------------------- | :--------------------------------------------------------------- |
| `src/etl_intra_match.py` | ETL segmentado por período; output: `intra_match_data.csv` |
| `src/features_inplay.py` | Calcula features de delta e combina com histórico pré-jogo |
| `src/train_inplay.py` | Treina M1, M2, M3 e gera gráfico comparativo de acurácia |

---

## **6. Critérios de Sucesso e Validação**

Para que a hipótese do orientador seja corroborada pela evidência empírica, ao menos uma das seguintes condições deve ser satisfeita ao final da Iteração 03:

1. **M2 (1º Tempo):** A marca de ~33% de acurácia deve ser superada, comprovando-se a validade preditiva dos atributos intramatch.
2. **M3 (Combinado)** deve superar **M1 (65%)**, demonstrando que o sinal intra-jogo agrega valor preditivo sobre o histórico pré-jogo.
3. A plotagem de *Feature Importance* deve confirmar que variáveis de "estado de jogo" (como `xg_period_1` ou `score_diff_period`) figuram entre as mais influentes no modelo vencedor.

Caso nenhum dos critérios seja satisfeito, o resultado em si configura uma contribuição acadêmica relevante: a demonstração de que, para o conjunto de dados e métricas utilizados, o sinal intra-jogo não supera o histórico acumulado, o que abre caminho para especulações sobre granularidade de dados e complexidade de sistemas dinâmicos em esportes de baixa pontuação.
