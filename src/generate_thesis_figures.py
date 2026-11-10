import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import shap
from sklearn.ensemble import RandomForestRegressor

# Garantir diretório criado
os.makedirs('outputs/plots', exist_ok=True)

# ==========================================
# FIGURA 1: SISTEMA DE COORDENADAS STATSBOMB
# ==========================================
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 6))

# Setas de Eixos (Estilo Limpo)
# Eixo X movido para a parte inferior (Y=85) para não sobrepor o título no topo
ax.annotate('', xy=(120, 85), xytext=(0, 85), annotation_clip=False, arrowprops=dict(arrowstyle='->', color='w', lw=2))
ax.text(60, 89, 'Eixo X (0 a 120)', color='white', ha='center', va='top', fontsize=12)

# Eixo Y na lateral esquerda
ax.annotate('', xy=(-5, 80), xytext=(-5, 0), annotation_clip=False, arrowprops=dict(arrowstyle='->', color='w', lw=2))
ax.text(-7, 40, 'Eixo Y (0 a 80)', color='white', va='center', ha='right', rotation='vertical', fontsize=12)

# Direção de ataque
ax.annotate('Direção de Ataque', xy=(90, 40), xytext=(30, 40),
            arrowprops=dict(arrowstyle='-|>', color='#e63946', lw=4, ls='--'), 
            color='#e63946', fontsize=14, ha='center', fontweight='bold')

# plt.title('Figura 1: Sistema de Coordenadas StatsBomb e Direção de Ataque', color='white', fontsize=14, pad=15)
fig.patch.set_facecolor('#22312b')
plt.savefig('outputs/plots/fig1_coordenadas.png', dpi=300, bbox_inches='tight', facecolor='#22312b')
plt.close()


# ==========================================
# FIGURA 2: CONCEITO DE FIELD TILT E ZONAS
# ==========================================
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 6))

import matplotlib.patches as patches

# Zonas do Field Tilt (Terço final de cada time)
rect_b = patches.Rectangle((pitch.dim.right - 40, pitch.dim.bottom), 40, pitch.dim.top - pitch.dim.bottom, linewidth=0, facecolor='#e63946', alpha=0.3)
ax.add_patch(rect_b)

rect_a = patches.Rectangle((pitch.dim.left, pitch.dim.bottom), 40, pitch.dim.top - pitch.dim.bottom, linewidth=0, facecolor='#457b9d', alpha=0.3)
ax.add_patch(rect_a)

# Simular passes nos terços finais (Ações de domínio territorial)
np.random.seed(42)
pitch.scatter(np.random.uniform(85, 115, 20), np.random.uniform(10, 70, 20), ax=ax, color='#e63946', edgecolors='white', s=60, label='Ações Time A (No Terço Final B)')
pitch.scatter(np.random.uniform(5, 35, 10), np.random.uniform(10, 70, 10), ax=ax, color='#457b9d', edgecolors='white', s=60, label='Ações Time B (No Terço Final A)')

ax.text(100, 40, 'Terço Final\nDefensivo Time B\n(X > 80)', color='white', ha='center', fontsize=11, fontweight='bold')
ax.text(20, 40, 'Terço Final\nDefensivo Time A\n(X < 40)', color='white', ha='center', fontsize=11, fontweight='bold')

fig.legend(loc='lower center', ncol=2, facecolor='#22312b', edgecolor='white', labelcolor='white', bbox_to_anchor=(0.5, -0.05))
# plt.title('Figura 2: Ilustração do Controle Territorial (Cálculo de Field Tilt)', color='white', fontsize=14, pad=15)
fig.patch.set_facecolor('#22312b')
plt.savefig('outputs/plots/fig2_field_tilt.png', dpi=300, bbox_inches='tight', facecolor='#22312b')
plt.close()


# ==========================================
# FIGURA 3: ABLAÇÃO SHAP E IMPORTÂNCIA
# ==========================================
# Criar dados genéricos baseados nas features do projeto (Iteração 2/3)
np.random.seed(42)
variaveis = ['xg_30_45', 'field_tilt_0_15', 'xg_15_30', 'field_tilt_15_30', 
             'xg_0_15', 'field_tilt_30_45', 'ppda_0_15', 'ppda_15_30']
X = pd.DataFrame(np.random.randn(200, len(variaveis)), columns=variaveis)
# Gerar um "target" fictício para que o SHAP gere o beeswarm adequadamente
y = X['xg_30_45'] * 1.5 + X['field_tilt_0_15'] * 1.2 + X['xg_15_30'] * -0.8 + np.random.randn(200) * 0.5 

# Treinar modelo dummy genérico para plot do Summary
model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
# Importante: show=False para não travar a execução do script
shap.summary_plot(shap_values, X, show=False)
# plt.title('Figura 3: Exemplo de Explicabilidade SHAP das Variáveis Temporais', pad=15)
plt.savefig('outputs/plots/fig3_shap_exemplo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("GERACAO_DE_IMAGENS_CONCLUIDA")
