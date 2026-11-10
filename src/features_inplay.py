import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def process_inplay_features(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File {input_path} not found. Run etl_intra_match.py first.")
        return

    # We need to reshape the dataframe. Right now each team-match has up to 2 rows (period 1 and period 2).
    # We want 1 row per team-match with period 1 features, period 2 features, and the momentum features.
    
    # Sort to ensure order
    df = df.sort_values(['match_id', 'team_name', 'period'])
    
    # Split periods
    df_p1 = df[df['period'] == 1].copy()
    df_p2 = df[df['period'] == 2].copy()
    
    # Rename columns for clarity
    p1_cols = {c: f"{c}_p1" for c in ['score_diff_period', 'field_tilt_period', 'xg_period', 'ppda_period', 'shots_on_target', 'deep_completions']}
    p2_cols = {c: f"{c}_p2" for c in ['score_diff_period', 'field_tilt_period', 'xg_period', 'ppda_period', 'shots_on_target', 'deep_completions']}
    
    df_p1 = df_p1.rename(columns=p1_cols)
    df_p2 = df_p2.rename(columns=p2_cols)
    
    # Merge periods
    merge_keys = ['match_id', 'match_date', 'competition_id', 'season_id', 'team_name', 'opponent_name', 'is_home', 'target_multiclass']
    df_merged = pd.merge(
        df_p1.drop(columns=['period']), 
        df_p2[merge_keys + list(p2_cols.values())], 
        on=merge_keys, 
        how='inner'
    )
    
    # Calculate Momentum Features (Delta = P2 - P1)
    df_merged['delta_xg'] = df_merged['xg_period_p2'] - df_merged['xg_period_p1']
    df_merged['delta_ppda'] = df_merged['ppda_period_p2'] - df_merged['ppda_period_p1']
    df_merged['delta_field_tilt'] = df_merged['field_tilt_period_p2'] - df_merged['field_tilt_period_p1']
    df_merged['delta_score'] = df_merged['score_diff_period_p2'] - df_merged['score_diff_period_p1']
    
    # Optional: Fill missing with 0 or mean
    df_merged = df_merged.fillna(0)
    
    # Sort by match_date
    df_merged = df_merged.sort_values('match_date')
    
    # Historical rolling averages (simulated from P1/P2/Total)
    # The prompt mentions using baseline Historical (M1). Let's calculate a rolling average of Total Match stats to use for M1.
    df_merged['total_xg'] = df_merged['xg_period_p1'] + df_merged['xg_period_p2']
    df_merged['total_ppda'] = (df_merged['ppda_period_p1'] + df_merged['ppda_period_p2']) / 2
    df_merged['total_field_tilt'] = (df_merged['field_tilt_period_p1'] + df_merged['field_tilt_period_p2']) / 2
    
    # Sort by team and date for rolling averages
    df_merged = df_merged.sort_values(['team_name', 'match_date'])
    
    window_size = 5
    historical_cols = ['total_xg', 'total_ppda', 'total_field_tilt']
    for col in historical_cols:
        df_merged[f'roll_{col}'] = df_merged.groupby('team_name')[col].transform(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        
    # Impute missing rolling with overall mean
    for col in historical_cols:
        df_merged[f'roll_{col}'] = df_merged[f'roll_{col}'].fillna(df_merged[col].mean())
        
    # Re-sort by match date
    df_merged = df_merged.sort_values('match_date').reset_index(drop=True)
    
    print(f"Generated In-Play Modeling Data with shape: {df_merged.shape}")
    df_merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    process_inplay_features('data/intra_match_data.csv', 'data/inplay_modeling_data.csv')
