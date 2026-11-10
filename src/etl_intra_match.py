import pandas as pd
import numpy as np
import os
from statsbombpy import sb
import requests_cache
import warnings
import time

# Redirecionando o cache do StatsBomb para o seu HD E: para evitar lotar o C:
cache_dir = os.path.join('_cache')
os.makedirs(cache_dir, exist_ok=True)
requests_cache.install_cache(os.path.join(cache_dir, 'statsbomb_cache'), backend='sqlite')

warnings.filterwarnings('ignore')

def calculate_ppda_period(events_df, team_name, period):
    df = events_df[events_df['period'] == period].copy()
    if df.empty:
        return np.nan
        
    df['x'] = df['location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan)
    df = df.dropna(subset=['x'])

    opponent_passes = df[
        (df['team'] != team_name) &
        (df['type'] == 'Pass') &
        (df['x'] <= 60)
    ]
    num_opp_passes = len(opponent_passes)

    defensive_actions = ['Duel', 'Interception', 'Ball Recovery', 'Block', 'Foul Committed', 'Pressure']
    team_def_actions = df[
        (df['team'] == team_name) &
        (df['type'].isin(defensive_actions)) & 
        (df['x'] > 60)
    ]
    num_def_actions = len(team_def_actions)

    if num_def_actions == 0:
        return 50.0 

    return num_opp_passes / num_def_actions

def calculate_deep_completions_period(events_df, team_name, period):
    df = events_df[events_df['period'] == period].copy()
    if df.empty:
        return 0
        
    completed_passes = df[
        (df['team'] == team_name) &
        (df['type'] == 'Pass') &
        (df['pass_outcome'].isna()) 
    ].copy()

    if completed_passes.empty:
        return 0

    completed_passes['end_x'] = completed_passes['pass_end_location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan)
    completed_passes['end_y'] = completed_passes['pass_end_location'].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) else np.nan)
    completed_passes = completed_passes.dropna(subset=['end_x', 'end_y'])

    in_box = (completed_passes['end_x'] >= 102) & (completed_passes['end_y'] >= 18) & (completed_passes['end_y'] <= 62)
    dist_to_goal = np.sqrt((completed_passes['end_x'] - 120)**2 + (completed_passes['end_y'] - 40)**2)
    within_20m = dist_to_goal <= 20

    deep_comps = completed_passes[in_box | within_20m]
    return len(deep_comps)

def calculate_xg_period(events_df, team_name, period):
    df = events_df[events_df['period'] == period].copy()
    team_shots = df[
        (df['team'] == team_name) &
        (df['type'] == 'Shot')
    ]
    if 'shot_statsbomb_xg' in team_shots.columns:
        return team_shots['shot_statsbomb_xg'].sum()
    return 0.0

def calculate_shots_on_target_period(events_df, team_name, period):
    df = events_df[events_df['period'] == period].copy()
    team_shots = df[
        (df['team'] == team_name) &
        (df['type'] == 'Shot')
    ]
    if 'shot_outcome' in team_shots.columns:
        sot = team_shots[team_shots['shot_outcome'].isin(['Goal', 'Saved', 'Saved to Post'])]
        return len(sot)
    return 0

def get_field_tilt_period(events_df, period):
    df = events_df[events_df['period'] == period].copy()
    if df.empty:
        return {}
        
    passes = df[df['type'] == 'Pass'].copy()
    passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else 0)
    final_third_passes = passes[passes['x'] >= 80]
    
    team_counts = final_third_passes['team'].value_counts()
    total_final_third = team_counts.sum()

    if total_final_third == 0:
        return {team: 0.0 for team in df['team'].unique()}

    field_tilt = (team_counts / total_final_third).to_dict()
    for team in df['team'].unique():
        if team not in field_tilt:
            field_tilt[team] = 0.0
    return field_tilt

def get_goals_in_period(events_df, team_name, period):
    df = events_df[events_df['period'] == period].copy()
    goals = df[
        (df['team'] == team_name) &
        (df['type'] == 'Shot') &
        (df['shot_outcome'] == 'Goal')
    ]
    return len(goals)

def process_match_periods(match_details, comp_id, season_id):
    match_id = match_details['match_id']
    home_team = match_details['home_team']
    away_team = match_details['away_team']

    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"Skipping match {match_id} due to API error: {e}")
        return []

    if events.empty or 'period' not in events.columns:
        return []

    rows = []
    
    for period in [1, 2]:
        field_tilt = get_field_tilt_period(events, period)
        
        home_goals_period = get_goals_in_period(events, home_team, period)
        away_goals_period = get_goals_in_period(events, away_team, period)
        
        home_score_diff = home_goals_period - away_goals_period
        away_score_diff = away_goals_period - home_goals_period
        
        home_score = match_details['home_score']
        away_score = match_details['away_score']
        
        target_mc_home = 2 if home_score > away_score else (1 if home_score == away_score else 0)
        target_mc_away = 2 if away_score > home_score else (1 if home_score == away_score else 0)

        for team, opponent, is_home, score_diff, target_multiclass in [
            (home_team, away_team, 1, home_score_diff, target_mc_home),
            (away_team, home_team, 0, away_score_diff, target_mc_away)
        ]:
            row = {
                'match_id': match_id,
                'match_date': match_details['match_date'],
                'competition_id': comp_id,
                'season_id': season_id,
                'team_name': team,
                'opponent_name': opponent,
                'is_home': is_home,
                'period': period,
                'target_multiclass': target_multiclass,
                'score_diff_period': score_diff,
                'field_tilt_period': field_tilt.get(team, 0.0),
                'xg_period': calculate_xg_period(events, team, period),
                'ppda_period': calculate_ppda_period(events, team, period),
                'shots_on_target': calculate_shots_on_target_period(events, team, period),
                'deep_completions': calculate_deep_completions_period(events, team, period)
            }
            rows.append(row)

    return rows

def etl_intra_match_pipeline(limit_matches=None):
    print("Fetching competitions...")
    comps = sb.competitions()

    # Use all competitions
    target_comps = comps
    print(f"Found {len(target_comps)} relevant seasons to process.")

    all_rows = []
    matches_processed = 0

    os.makedirs('data', exist_ok=True)

    for _, comp in target_comps.iterrows():
        comp_id = comp['competition_id']
        season_id = comp['season_id']
        comp_name = comp['competition_name']
        season_name = comp['season_name']

        print(f"\nProcessing {comp_name} - {season_name}...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            continue

        if limit_matches:
            matches = matches.head(limit_matches)

        for _, match in matches.iterrows():
            if matches_processed % 50 == 0:
                print(f"  -> Processed {matches_processed} matches...")
            match_rows = process_match_periods(match, comp_id, season_id)
            all_rows.extend(match_rows)

            matches_processed += 1
            if limit_matches and matches_processed >= limit_matches:
                break
        if limit_matches and matches_processed >= limit_matches:
            break

    print("\nETL COMPLETE.")
    final_df = pd.DataFrame(all_rows)
    print(f"Total rows generated: {len(final_df)} ")
    
    output_path = 'data/intra_match_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    start_time = time.time()
    etl_intra_match_pipeline(limit_matches=None)
    print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")
