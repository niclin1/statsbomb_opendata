import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(df, features, target_col, model_name):
    # Sort for time-series split
    df = df.sort_values('match_date')
    X = df[features]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))

    from sklearn.metrics import f1_score
    preds_full = model.predict(X)
    f1_macro = f1_score(y, preds_full, average='macro')
    mean_acc = np.mean(accuracies)
    print(f"--- {model_name} ---")
    print(f"TSCV Accuracy: {mean_acc:.2%}")
    print(f"Full F1-Macro:  {f1_macro:.2%}")
    print(f"Mean TimeSeries CV Accuracy: {mean_acc:.4f}\n")
    return model, mean_acc, X_train.columns

def run_experiment():
    print("Loading data...")
    df = pd.read_csv('data/inplay_modeling_data.csv')

    target = 'target_multiclass'

    # Model 1: Pre-match history only (Rolling averages)
    m1_features = ['is_home', 'roll_total_xg', 'roll_total_ppda', 'roll_total_field_tilt']

    # Model 2: 1st Half only (In-Play)
    m2_features = [
        'is_home', 'score_diff_period_p1', 'xg_period_p1', 
        'ppda_period_p1', 'field_tilt_period_p1', 
        'shots_on_target_p1', 'deep_completions_p1'
    ]

    # Model 3: Combined (Pre-match + In-Play)
    m3_features = m1_features + m2_features
    # remove duplicates
    m3_features = list(dict.fromkeys(m3_features))

    results = {}

    m1_model, m1_acc, m1_cols = train_and_evaluate(df, m1_features, target, "M1 - Pre-Match Baseline")
    results['M1 (Pre-Match Baseline)'] = m1_acc

    m2_model, m2_acc, m2_cols = train_and_evaluate(df, m2_features, target, "M2 - 1st Half Only")
    results['M2 (1st Half Only)'] = m2_acc

    m3_model, m3_acc, m3_cols = train_and_evaluate(df, m3_features, target, "M3 - Combined")
    results['M3 (Combined)'] = m3_acc

    # Plot Accuracy Comparison
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green'])
    plt.ylabel('Accuracy')
    plt.title('Iteration 03: Feature Paradigms Comparison (Accuracy)')
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.1%}", ha='center')
    plt.savefig('outputs/plots/iteration_03_accuracy_comparison.png')
    plt.close()
    
    # Plot Feature Importance for M3
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(m3_model, max_num_features=10, importance_type='gain', title='M3 Combined Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('outputs/plots/iteration_03_m3_feature_importance.png')
    plt.close()

    print("Experiment completed. Outputs saved to outputs/plots/.")
    return results

if __name__ == '__main__':
    run_experiment()
