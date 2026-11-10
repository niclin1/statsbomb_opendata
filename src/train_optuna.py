import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def load_data():
    try:
        df = pd.read_csv('data/momentum_data.csv')
    except FileNotFoundError:
        print("Arquivo momentum_data.csv não encontrado. Aguarde o fim do ETL.")
        return None, None, None

    df['match_date'] = pd.to_datetime(df['match_date'])
    # Garantir a ordenação temporal estrita para o Walk-Forward Validation
    df = df.sort_values(by='match_date').reset_index(drop=True)
    
    # Isolar Features (X) e Target Multiclasse (y)
    exclude_cols = ['match_id', 'match_date', 'competition_id', 'season_id', 'team_name', 'opponent_name', 'is_home', 'target_multiclass']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y = df['target_multiclass']
    return X, y, features

def objective(trial, X, y):
    # Dicionário de hiperparâmetros para XGBoost sugerido pelo Optuna
    param = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'eta': trial.suggest_float('eta', 1e-3, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }

    # Cross-Validation focado em Séries Temporais (Múltiplas Fatias do Tempo)
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.argmax(preds, axis=1)
        accuracies.append(accuracy_score(y_test, pred_labels))

    # Optuna tentará MAximizar esta média de acurácia nos 5 splits de tempo
    return np.mean(accuracies)

def train_and_evaluate():
    X, y, features = load_data()
    if X is None: return
    
    print("Iniciando Otimização Bayesiana Multiclasse com Optuna...")
    # Criar estudo do Optuna focado em MAXIMIZAR o Accuracy
    study = optuna.create_study(direction='maximize', study_name='XGBoost_Momentum')
    # Otimizando através de 50 tentativas (Trials)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=1)

    print("\n==============================")
    print("MELHOR RESULTADO (TRIAL):")
    trial = study.best_trial
    print(f"Acurácia Média no Tempo: {trial.value:.4f}")
    print("Melhores Hiperparâmetros encontrados:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print("==============================\n")

    # ==========================
    # TREINAMENTO DO MODELO FINAL COM OS DADOS DE TESTE (20% mais recentes)
    # ==========================
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    final_params = trial.params
    final_params['objective'] = 'multi:softprob'
    final_params['num_class'] = 3
    final_params['eval_metric'] = 'mlogloss'

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Treina novamente com a receita perfeita
    bst = xgb.train(final_params, dtrain)
    preds = bst.predict(dtest)
    pred_labels = np.argmax(preds, axis=1)

    print("Relatório de Classificação Final do 20% do Futuro (Conjunto de Teste Extremo):")
    print(classification_report(y_test, pred_labels, target_names=['Derrota (0)', 'Empate (1)', 'Vitória (2)']))

    # ==========================
    # IMPORTÂNCIA VETORIAL (XAI / SHAP VALUES)
    # ==========================
    print("Gerando Extração Computacional Nativa SHAP para a Monografia...")
    
    # O TreeExplainer lê o XGBoost matematicamente de trás pra frente
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
    
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # shap_values[2] = Isola os pesos preditivos apenas para a Classe 2 (VITÓRIA)
    # Ou seja: Quais métricas de Momento aumentam a chance matemática da Vitória?
    if isinstance(shap_values, list):
        shap_values_target = shap_values[2]
    else:
        # Nas versões recentes do xgboost/shap (array 3D: amostras x features x classes)
        shap_values_target = shap_values[:, :, 2]
        
    shap.summary_plot(shap_values_target, X_test, feature_names=features, show=False)
    
    plt.title("Ablação SHAP - Importância Nativa das Variáveis para Previsão de Vitória")
    plt.tight_layout()
    plot_path = 'outputs/plots/optuna_shap_win_class.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico Final SHAP salvo com sucesso em: {plot_path}")

if __name__ == "__main__":
    train_and_evaluate()