import pandas as pd
import numpy as np
import os
from statsbombpy import sb
import warnings
import time
from traceback import print_exc

warnings.filterwarnings('ignore')

# Helper functions for the new metrics
def calculate_ppda(events_df, team_name):
    """
    Passes Per Defensive Action (PPDA)
    (Passes do adversário na defesa) / (Ações defensivas no ataque)
    """
    df = events_df.copy()
    
    # Extract x coordinate
    df['x'] = df['location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan)
    df = df.dropna(subset=['x'])
    
    # Opponent passes in their defensive half (x <= 60 for the opponent)
    opponent_passes = df[
        (df['team'] != team_name) & 
        (df['type'] == 'Pass') & 
        (df['x'] <= 60)
    ]
    num_opp_passes = len(opponent_passes)
    
    # Active team's defensive actions in their attacking half (x > 60)
    # StatsBomb defensive actions standard definition:
    # Duel, Interception, Ball Recovery, Block, Clearance (sometimes), Pressure
    # Let's use the core defensive acts:
    defensive_actions = ['Duel', 'Interception', 'Ball Recovery', 'Block', 'Foul Committed', 'Pressure']
    
    team_def_actions = df[
        (df['team'] == team_name) & 
        (df['type'].isin(defensive_actions)) & 
        (df['x'] > 60)
    ]
    num_def_actions = len(team_def_actions)
    
    # PPDA Calculation (handle division by zero)
    if num_def_actions == 0:
        return np.nan # Or a very high number like 50.0, but NaN lets us impute later or just signifies no pressing. Let's use 50.0 as a max cap.
        
    return num_opp_passes / num_def_actions

def calculate_deep_completions(events_df, team_name):
    """
    Passes completados dentro de um raio de 20 metros do gol ou dentro da área penal.
    Gol X: 120. Y: 40.
    Raio de 20m: (x - 120)^2 + (y - 40)^2 <= 20^2
    Or simply within the box (X >= 102, Y between 18 and 62).
    """
    df = events_df.copy()
    
    # Filter completed passes for the team
    completed_passes = df[
        (df['team'] == team_name) & 
        (df['type'] == 'Pass') & 
        (df['pass_outcome'].isna()) # NaN means completed in SB
    ].copy()
    
    completed_passes['end_x'] = completed_passes['pass_end_location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else np.nan)
    completed_passes['end_y'] = completed_passes['pass_end_location'].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) else np.nan)
    
    completed_passes = completed_passes.dropna(subset=['end_x', 'end_y'])
    
    # Definition 1: Inside the Box
    in_box = (completed_passes['end_x'] >= 102) & (completed_passes['end_y'] >= 18) & (completed_passes['end_y'] <= 62)
    
    # Definition 2: Within 20m of center of goal (120, 40)
    dist_to_goal = np.sqrt((completed_passes['end_x'] - 120)**2 + (completed_passes['end_y'] - 40)**2)
    within_20m = dist_to_goal <= 20
    
    # Combine (OR condition is often used, but usually just being near the goal counts)
    deep_comps = completed_passes[in_box | within_20m]
    
    return len(deep_comps)

def calculate_xg(events_df, team_name):
    """Total expected goals for the team."""
    team_shots = events_df[
        (events_df['team'] == team_name) & 
        (events_df['type'] == 'Shot')
    ]
    if 'shot_statsbomb_xg' in team_shots.columns:
        return team_shots['shot_statsbomb_xg'].sum()
    return 0.0

def get_field_tilt_fast(events_df):
    """Faster Field Tilt calculation for the whole match."""
    df = events_df.copy()
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


def process_match(match_details, comp_id, season_id):
    """Processes a single match and returns two rows of data (Home and Away)."""
    match_id = match_details['match_id']
    home_team = match_details['home_team']
    away_team = match_details['away_team']
    
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"Skipping match {match_id} due to API error: {e}")
        return []
        
    if events.empty:
        return []
        
    # Pre-calculate Field Tilt to avoid repeating
    field_tilt = get_field_tilt_fast(events)
    
    rows = []
    
    # Determine Result for Home Team (0 = Win, 1 = Draw, 2 = Loss)
    home_score = match_details['home_score']
    away_score = match_details['away_score']
    
    home_result = 0 if home_score > away_score else (1 if home_score == away_score else 2)
    away_result = 2 if home_result == 0 else (1 if home_result == 1 else 0)
    # Using 1 for Win, 0 for Draw/Loss makes binary classification easier later, 
    # but the prompt asked for multi-class target_win (1 if won, 0 if not). Let's use 1 = Win, 0 = Draw/Loss
    target_win_home = 1 if home_score > away_score else 0
    target_win_away = 1 if away_score > home_score else 0

    for team, opponent, is_home, goals_for, goals_against, target_win in [
        (home_team, away_team, 1, home_score, away_score, target_win_home),
        (away_team, home_team, 0, away_score, home_score, target_win_away)
    ]:
        row = {
            'match_id': match_id,
            'match_date': match_details['match_date'],
            'competition_id': comp_id,
            'season_id': season_id,
            'team_name': team,
            'opponent_name': opponent,
            'is_home': is_home,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'target_win': target_win,
            'field_tilt': field_tilt.get(team, 0.0),
            'ppda': calculate_ppda(events, team),
            'total_xg': calculate_xg(events, team),
            'deep_completions': calculate_deep_completions(events, team)
        }
        rows.append(row)
        
    return rows

def etl_pipeline(limit_matches=None):
    print("Fetching competitions...")
    comps = sb.competitions()
    
    # Filter for La Liga (comp_id 11) and World Cups (comp_id 43)
    target_comps = comps[comps['competition_id'].isin([11, 43])]
    print(f"Found {len(target_comps)} relevant seasons to process.")
    
    all_rows = []
    matches_processed = 0
    
    # Create outputs directory
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
             print(f"Failed to fetch matches for {comp_name} - {season_name}: {e}")
             continue
             
        # Optional limit for testing
        if limit_matches:
            matches = matches.head(limit_matches)
             
        for _, match in matches.iterrows():
            print(f"  -> Match {match['match_id']}: {match['home_team']} vs {match['away_team']}")
            
            match_rows = process_match(match, comp_id, season_id)
            all_rows.extend(match_rows)
            
            matches_processed += 1
            
            # Save intermediate progress every 50 matches just in case
            if matches_processed % 50 == 0:
                print(f"Checkpoint reached. Saving intermediate data ({matches_processed} matches processed)...")
                temp_df = pd.DataFrame(all_rows)
                temp_df.to_csv('data/historical_match_data_checkpoint.csv', index=False)
                
            if limit_matches and matches_processed >= limit_matches:
                break
                
        if limit_matches and matches_processed >= limit_matches:
            break
            
    print("\nETL COMPLETE.")
    final_df = pd.DataFrame(all_rows)
    print(f"Total rows generated: {len(final_df)} (representing {len(final_df)//2} matches)")
    
    output_path = 'data/historical_match_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Remove limit_matches to process all data. 
    # For this test, I will set a limit of 20 matches per competition just to verify the pipeline logic
    # without taking 5 hours to run. The user can remove the limit later.
    
    # Actually, I'll limit to 30 matches total so the test finishes reasonably fast
    # but generates enough data to inspect.
    print("Starting Phase 1: Data Scaling & ETL (Full Extration)")
    start_time = time.time()
    etl_pipeline() 
    print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")
