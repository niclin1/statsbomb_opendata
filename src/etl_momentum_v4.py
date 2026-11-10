import pandas as pd
import numpy as np
import os
import time
import warnings
from statsbombpy import sb
import requests_cache

# Redirecionando o cache do StatsBomb para o diretório local (Drive E:) e evitando lotar o C:
cache_dir = os.path.join('_cache')
os.makedirs(cache_dir, exist_ok=True)
requests_cache.install_cache(os.path.join(cache_dir, 'statsbomb_cache'), backend='sqlite')

warnings.filterwarnings('ignore')

def is_red_card(event):
    if event.get('type') == 'Foul Committed' and event.get('foul_committed_card') in ['Red Card', 'Second Yellow']:
        return True
    if event.get('type') == 'Bad Behaviour' and event.get('bad_behaviour_card') in ['Red Card', 'Second Yellow']:
        return True
    return False

def calculate_ppda_segment(df, team_name):
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

def calculate_xg_segment(df, team_name):
    team_shots = df[
        (df['team'] == team_name) &
        (df['type'] == 'Shot')
    ]
    if 'shot_statsbomb_xg' in team_shots.columns:
        return team_shots['shot_statsbomb_xg'].sum()
    return 0.0

def get_field_tilt_segment(df):
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

def process_match_momentum(match_details, comp_id, season_id):
    match_id = match_details['match_id']
    home_team = match_details['home_team']
    away_team = match_details['away_team']

    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        return []

    if events.empty or 'period' not in events.columns or 'minute' not in events.columns:
        return []

    # Filter only 1st Half for Momentum logic
    df_p1 = events[events['period'] == 1].copy()
    
    # 15-Minute Bins
    df_0_15 = df_p1[df_p1['minute'] < 15]
    df_15_30 = df_p1[(df_p1['minute'] >= 15) & (df_p1['minute'] < 30)]
    df_30_45 = df_p1[df_p1['minute'] >= 30]

    # Red Cards in 1st Half
    red_cards = df_p1[df_p1.apply(is_red_card, axis=1)]
    home_reds = len(red_cards[red_cards['team'] == home_team])
    away_reds = len(red_cards[red_cards['team'] == away_team])

    # Calculate metrics per segment
    segments = [("0_15", df_0_15), ("15_30", df_15_30), ("30_45", df_30_45)]
    
    team_metrics = {home_team: {}, away_team: {}}
    
    for seg_name, seg_df in segments:
        ft = get_field_tilt_segment(seg_df)
        for t in [home_team, away_team]:
            team_metrics[t][f'xg_{seg_name}'] = calculate_xg_segment(seg_df, t)
            team_metrics[t][f'ppda_{seg_name}'] = calculate_ppda_segment(seg_df, t)
            team_metrics[t][f'field_tilt_{seg_name}'] = ft.get(t, 0.0)

    # Target class (Multiclass: 0=Loss, 1=Draw, 2=Win)
    if match_details['home_score'] > match_details['away_score']:
        target_home, target_away = 2, 0
    elif match_details['home_score'] < match_details['away_score']:
        target_home, target_away = 0, 2
    else:
        target_home, target_away = 1, 1

    rows = []
    for team, opponent, is_home, target_mc, team_reds, opp_reds in [
        (home_team, away_team, 1, target_home, home_reds, away_reds),
        (away_team, home_team, 0, target_away, away_reds, home_reds)
    ]:
        row = {
            'match_id': match_id,
            'match_date': match_details['match_date'],
            'competition_id': comp_id,
            'season_id': season_id,
            'team_name': team,
            'opponent_name': opponent,
            'is_home': is_home,
            'target_multiclass': target_mc,
            'team_red_cards_p1': team_reds,
            'opp_red_cards_p1': opp_reds
        }
        row.update(team_metrics[team])
        rows.append(row)

    return rows

def etl_momentum_pipeline(limit_matches=None):
    print("Initiating Momentum & Micro-segments ETL...")
    comps = sb.competitions()
    
    # Reduzindo o escopo para ligas táticas modernas (Champions League e Copas Recentes)
    target_comps = comps[
        # Removemos jogos "soltos" do passado e garantimos volume grande
        (comps['competition_name'].isin(['La Liga', 'Premier League', 'Serie A', 'Ligue 1', '1. Bundesliga', 'FIFA World Cup', 'UEFA Euro', 'Copa America', 'Indian Super league'])) &
        (comps['season_name'].str.startswith('202') | comps['season_name'].str.startswith('201'))
    ]
    
    print(f"Found {len(target_comps)} relevant modern tactical seasons to process.")

    all_rows = []
    matches_processed = 0

    os.makedirs('data', exist_ok=True)

    for _, comp in target_comps.iterrows():
        comp_id = comp['competition_id']
        season_id = comp['season_id']
        
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:
            continue

        if limit_matches:
            matches = matches.head(limit_matches)

        for _, match in matches.iterrows():
            if matches_processed % 100 == 0:
                print(f"Processed {matches_processed} matches...")
                
            match_rows = process_match_momentum(match, comp_id, season_id)
            all_rows.extend(match_rows)

            matches_processed += 1
            if limit_matches and matches_processed >= limit_matches:
                break
        if limit_matches and matches_processed >= limit_matches:
            break

    print("\nETL MOMENTUM COMPLETE.")
    final_df = pd.DataFrame(all_rows)
    print(f"Total rows generated: {len(final_df)} ")
    
    output_path = 'data/momentum_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    start_time = time.time()
    etl_momentum_pipeline(limit_matches=None)
    print(f"Execution time: {(time.time() - start_time) / 60:.2f} minutes")