import pandas as pd
import numpy as np
import os

def create_time_series_features(input_file, output_file, window=5):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Initial shape: {df.shape}")

    # Ensure date is string/datetime for sorting
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Sort chronologically
    df = df.sort_values(by=['team_name', 'match_date']).reset_index(drop=True)
    
    # Metrics to calculate rolling averages for
    metrics = ['goals_for', 'goals_against', 'field_tilt', 'ppda', 'total_xg', 'deep_completions']
    
    # Calculate rolling averages for the TEAM
    print(f"Calculating {window}-game rolling averages for teams...")
    for metric in metrics:
        col_name = f'team_recent_{metric}'
        # shift(1) ensures we don't include the current match in the "recent form"
        df[col_name] = df.groupby('team_name')[metric].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=window).mean()
        )
    
    # We also need the "Opponent Form" for each row.
    # To do this, we create a lookup dataframe of team forms at any given match_date
    form_lookup = df[['match_id', 'team_name'] + [f'team_recent_{m}' for m in metrics]].copy()
    
    # Rename for merging as opponent
    rename_cols = {'team_name': 'opponent_name'}
    for m in metrics:
        rename_cols[f'team_recent_{m}'] = f'opp_recent_{m}'
    form_lookup.rename(columns=rename_cols, inplace=True)
    
    print("Merging opponent recent form...")
    # Merge the opponent's recent form back into the main dataframe
    df = df.merge(form_lookup, on=['match_id', 'opponent_name'], how='left')
    
    # Drop rows with NaN values (these are the first 5 games of each team where they have no history)
    print(f"Dropping warm-up matches (first {window} games per team)...")
    final_df = df.dropna().reset_index(drop=True)
    
    print(f"Final shape for modeling: {final_df.shape}")
    
    # Select final features
    feature_cols = [
        'match_id', 'match_date', 'competition_id', 'season_id',
        'team_name', 'opponent_name', 'is_home', 'target_multiclass'
    ]
    # Add rolling features
    feature_cols += [f'team_recent_{m}' for m in metrics]
    feature_cols += [f'opp_recent_{m}' for m in metrics]
    
    final_df = final_df[feature_cols]
    
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved modeling data to {output_file}")

if __name__ == "__main__":
    # The user mentioned they stopped the ETL. Thus, they likely have a larger checkpoint file 
    # instead of the full historical_match_data.csv. Let's try the checkpoint if it exists and is larger.
    # Actually, we will just use whatever has more rows.
    
    # Check sizes
    size_main = os.path.getsize('data/historical_match_data.csv') if os.path.exists('data/historical_match_data.csv') else 0
    size_ckpt = os.path.getsize('data/historical_match_data_checkpoint.csv') if os.path.exists('data/historical_match_data_checkpoint.csv') else 0
    
    input_csv = 'data/historical_match_data_checkpoint.csv' if size_ckpt > size_main else 'data/historical_match_data.csv'
    
    create_time_series_features(
        input_file=input_csv,
        output_file='data/modeling_data.csv',
        window=5
    )
