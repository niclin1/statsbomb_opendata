import pandas as pd
import numpy as np
from statsbombpy import sb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from src.features_spatial import calculate_zone_metrics
from src.features_advanced import calculate_field_tilt, calculate_team_convex_hull

warnings.filterwarnings('ignore')

def get_match_metrics(match_id, home_team, away_team):
    """
    Extracts metrics for a single match for both teams.
    """
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"Error fetching events for match {match_id}: {e}")
        return None

    # Calculate Raw Metrics
    spatial_metrics = calculate_zone_metrics(events) # Total for match (not split by team yet in that func? Wait, calculate_zone_metrics returns totals? 
    # Checking calculate_zone_metrics implementation... 
    # It returns len(filtered_df). It does NOT split by team.
    # We need to adapt logic to split by team.
    # The current implementation of calculate_zone_metrics aggregates the WHOLE match.
    # I will do team splitting here before calling metrics functions or filter df first.
    
    # Better approach: filter events by team here and pass to metric calculators?
    # But calculate_field_tilt handles teams inside.
    # calculate_team_convex_hull handles teams inside.
    # calculate_zone_metrics needs a filtered DF to give per-team stats.

    metrics = {home_team: {}, away_team: {}}
    
    # 1. Advanced Metrics (Handled internally per team)
    ft = calculate_field_tilt(events)
    hull = calculate_team_convex_hull(events)
    
    metrics[home_team]['Field_Tilt'] = ft.get(home_team, 0.0)
    metrics[away_team]['Field_Tilt'] = ft.get(away_team, 0.0)
    
    metrics[home_team]['Hull_Area'] = hull.get(home_team, 0.0)
    metrics[away_team]['Hull_Area'] = hull.get(away_team, 0.0)
    
    # 2. Spatial Metrics (Need to filter DF)
    for team in [home_team, away_team]:
        team_events = events[events['team'] == team]
        zone_metrics = calculate_zone_metrics(team_events)
        metrics[team]['Zone14_Passes'] = zone_metrics['zone_14_passes']
        metrics[team]['Box_Shots'] = zone_metrics['box_shots']
        metrics[team]['Half_Space_Actions'] = zone_metrics['offensive_half_space_actions']
        
    return metrics

def build_dataset(competition_id=43, season_id=3): # WC 2018
    print(f"Fetching matches for Competition {competition_id}, Season {season_id}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches = matches.sort_values('match_date')
    
    match_data = []
    
    total_matches = len(matches)
    print(f"Found {total_matches} matches. Processing...")
    
    for idx, row in matches.iterrows():
        match_id = row['match_id']
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        date = row['match_date']
        
        print(f"[{idx+1}/{total_matches}] Processing {home_team} vs {away_team}...")
        
        metrics = get_match_metrics(match_id, home_team, away_team)
        if not metrics:
            continue
            
        # Determine Result (0: Home, 1: Draw, 2: Away) - useful for final target
        # For individual team Rows: 
        # We want to predict Outcome from Team's perspective? 
        # Or predict Match Result (Home/Draw/Away) based on HomeStats vs AwayStats?
        # Standard approach: Predict Match Result.
        
        if home_score > away_score:
            result = 0 # Home Win
        elif home_score == away_score:
            result = 1 # Draw
        else:
            result = 2 # Away Win
            
        # Append rows for feature engineering (we need a row per team per date to calculate rolling avgs)
        # Row for Home Team
        match_data.append({
            'date': date,
            'match_id': match_id,
            'team': home_team,
            'opponent': away_team,
            'is_home': 1,
            'goals_for': home_score,
            'goals_against': away_score,
            'result': result, # This matches the match result
            **metrics[home_team]
        })
        
        # Row for Away Team
        match_data.append({
            'date': date,
            'match_id': match_id,
            'team': away_team,
            'opponent': home_team,
            'is_home': 0,
            'goals_for': away_score,
            'goals_against': home_score,
            'result': result, # Same result target
            **metrics[away_team]
        })
        
    return pd.DataFrame(match_data)

def engineer_features(df):
    print("\nEngineering Rolling Features...")
    df = df.sort_values(['team', 'date'])
    
    metrics_cols = ['Field_Tilt', 'Hull_Area', 'Zone14_Passes', 'Box_Shots', 'Half_Space_Actions']
    
    # Calculate Rolling Average for the PREVIOUS 5 matches
    # shift(1) ensures we don't use the current match's stats to predict itself
    rolling_stats = df.groupby('team')[metrics_cols].apply(
        lambda x: x.shift(1).rolling(window=5, min_periods=3).mean() # min_periods=3 to save some early data
    ).reset_index(level=0, drop=True)
    
    # Rename columns
    rolling_stats.columns = [f'rolling_{col}' for col in metrics_cols]
    
    # Concatenate back
    df_rolling = pd.concat([df, rolling_stats], axis=1)
    
    # Drop rows without sufficient history
    df_final = df_rolling.dropna(subset=rolling_stats.columns)
    
    print(f"Dataset size after rolling window: {len(df_final)} rows (Teams * Matches)")
    return df_final

def prepare_match_level_dataset(df_rolling):
    """
    Reconstructs Match-Level rows combining Home Rolling Stats vs Away Rolling Stats.
    """
    # Split into Home and Away rows
    home_rows = df_rolling[df_rolling['is_home'] == 1].copy()
    away_rows = df_rolling[df_rolling['is_home'] == 0].copy()
    
    # Merge on match_id
    # Suffixes: _home, _away
    dataset = pd.merge(
        home_rows, 
        away_rows, 
        on=['match_id', 'date', 'result'], 
        suffixes=('_home', '_away')
    )
    
    return dataset

def train_model(dataset):
    print("\nTraining Baseline Model...")
    
    # Features: Rolling stats for Home and Away
    feature_cols = [c for c in dataset.columns if 'rolling_' in c]
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X = dataset[feature_cols]
    y = dataset['result']
    
    # Chronological Split
    # Sort just in case
    dataset = dataset.sort_values('date')
    
    total_samples = len(dataset)
    train_size = int(total_samples * 0.75)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=[0, 1, 2], target_names=['Home Win', 'Draw', 'Away Win']))
    
    return model

if __name__ == "__main__":
    # Use WC 2018
    raw_df = build_dataset(competition_id=43, season_id=3)
    
    if raw_df.empty:
        print("No data found!")
    else:
        df_rolling = engineer_features(raw_df)
        final_dataset = prepare_match_level_dataset(df_rolling)
        
        if len(final_dataset) < 10:
            print("Not enough data for training after rolling window.")
        else:
            train_model(final_dataset)
