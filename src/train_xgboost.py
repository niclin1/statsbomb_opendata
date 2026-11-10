import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/metrics', exist_ok=True)

def temporal_split(df, test_size_ratio=0.25):
    """Splits data temporally (oldest 75% train, newest 25% test)"""
    df_sorted = df.sort_values(by='match_date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size_ratio))
    
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    
    # Let's print the time boundary
    print(f"Train period: {train['match_date'].min()} to {train['match_date'].max()}")
    print(f"Test period:  {test['match_date'].min()} to {test['match_date'].max()}")
    
    return train, test

def train_and_evaluate():
    print("--- Phase 4: Modeling & Evaluation ---")
    data_path = 'data/modeling_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # 1. Temporal Split
    train, test = temporal_split(df)
    
    # Define features (all rolling numeric columns)
    features = [col for col in df.columns if 'recent_' in col or col == 'is_home']
    print(f"Features used ({len(features)}): {features}")
    
    X_train = train[features]
    y_train = train['target_multiclass']

    X_test = test[features]
    y_test = test['target_multiclass']

    # 2. Train Models
    # Baseline: Random Forest
    print("\nTraining Random Forest (Baseline)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)

    # Challenger: XGBoost
    print("Training XGBoost (Challenger)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        random_state=42,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)

    from sklearn.metrics import f1_score, roc_auc_score

    # 3. Evaluation
    def evaluate_model(name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        # roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr') # Some targets might not have all classes in small sets
        try:
            loss = log_loss(y_true, y_prob, labels=[0, 1, 2])
        except Exception:
            loss = float('nan')
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n[{name}] Results:")
        print(f"Accuracy:  {acc:.2%}")
        print(f"F1-Macro:  {f1_mac:.2%}")
        # print(f"ROC-AUC OVR: {roc_auc:.4f}")
        print(f"Log-Loss:  {loss:.4f}")
        print("Confusion Matrix:\n", cm)
        return acc, f1_mac, loss

    evaluate_model("Random Forest", y_test, rf_preds, rf_probs)
    evaluate_model("XGBoost", y_test, xgb_preds, xgb_probs)    # 4. Explainability (Feature Importance for XGBoost)
    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.figure(figsize=(10, 8))
    plt.barh(pos, importance[sorted_idx], align='center')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('outputs/plots/xgboost_feature_importance.png', dpi=300)
    print("\nFeature Importance plot saved to outputs/plots/xgboost_feature_importance.png")
    
    # 5. Error Analysis
    print("\n--- Error Analysis ---")
    # Where was the model most confident but wrong?
    test_eval = test.copy()
    test_eval['xgb_prob_max'] = np.max(xgb_probs, axis=1)
    test_eval['xgb_pred'] = xgb_preds
    test_eval['error'] = test_eval['target_multiclass'] != test_eval['xgb_pred']
    
    # Calculate confidence error magnitude (Probability of predicted class vs Truth)
    test_eval['confidence_error'] = test_eval['xgb_prob_max']

    worst_mistakes = test_eval[test_eval['error']].sort_values(by='confidence_error', ascending=False).head(3)

    print("Top 3 partidas com maior 'certeza no erro' pelo XGBoost:")
    report_lines = []

    for _, row in worst_mistakes.iterrows():
        prediction = 'Vitória' if row['xgb_pred'] == 2 else ('Empate' if row['xgb_pred'] == 1 else 'Derrota')
        truth = 'Vitória' if row['target_multiclass'] == 2 else ('Empate' if row['target_multiclass'] == 1 else 'Derrota')
        conf = row['confidence_error'] * 100

        msg = (f"[Data: {row['match_date']}] {row['team_name']} vs {row['opponent_name']} | "
               f"Previsão: {prediction} ({conf:.1f}%) | Realidade: {truth}\n"
               f"  -> Contexto: Team xG Recente ({row['team_recent_total_xg']:.2f}) vs Opp xG ({row['opp_recent_total_xg']:.2f}) | "
               f"Team PPDA ({row['team_recent_ppda']:.1f})")
        print(msg)
        report_lines.append(msg)
        
    with open('outputs/metrics/model_evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("RELATORIO DE AVALIACAO DE MODELOS (TCC)\n=======================================\n")
        f.write("\nErros de Confianca Maxima do XGBoost:\n")
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    train_and_evaluate()
