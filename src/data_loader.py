import pandas as pd
from statsbombpy import sb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_match_events(match_id):
    """
    Fetches and returns the events for a specific match.
    
    Args:
        match_id (int): The ID of the match.
        
    Returns:
        pd.DataFrame: A DataFrame containing the match events.
    """
    print(f"Fetching events for match_id: {match_id}...")
    events = sb.events(match_id=match_id)
    return events

def main():
    print("--- StatsBomb Data Loader ---")
    
    # 1. List Competitions
    print("\n1. Available Competitions (First 5):")
    competitions = sb.competitions()
    print(competitions.head())
    
    # 2. Find World Cup 2018
    print("\n2. Searching for 2018 World Cup...")
    wc_2018 = competitions[
        (competitions['competition_name'] == 'FIFA World Cup') & 
        (competitions['season_name'] == '2018')
    ]
    
    if wc_2018.empty:
        print("World Cup 2018 not found in competitions.")
        return

    competition_id = wc_2018.iloc[0]['competition_id']
    season_id = wc_2018.iloc[0]['season_id']
    print(f"Found: Competition ID {competition_id}, Season ID {season_id}")
    
    # 3. Get Matches for World Cup 2018
    print("\n3. Fetching Matches...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Assuming the final is the last match or finding it by stage
    # 2018 Final was France vs Croatia
    final = matches[
        ((matches['home_team'] == 'France') & (matches['away_team'] == 'Croatia')) |
        ((matches['home_team'] == 'Croatia') & (matches['away_team'] == 'France'))
    ]
    
    if final.empty:
        print("Final match not found, selecting the first available match.")
        match_to_load = matches.iloc[0]
    else:
        print("Found 2018 World Cup Final.")
        match_to_load = final.iloc[0]
        
    match_id = match_to_load['match_id']
    print(f"Selected Match: {match_to_load['home_team']} vs {match_to_load['away_team']} (ID: {match_id})")
    
    # 4. Get Match Events
    print("\n4. Fetching Match Events...")
    events = get_match_events(match_id)
    
    # 5. Spatial Feature Engineering
    print("\n5. Calculating Spatial Metrics...")
    try:
        from src.features_spatial import calculate_zone_metrics
    except ImportError:
        from features_spatial import calculate_zone_metrics
    
    metrics = calculate_zone_metrics(events)
    
    print("-" * 30)
    print(f"Zone 14 Passes: {metrics['zone_14_passes']}")
    print(f"Box Shots: {metrics['box_shots']}")
    print(f"Offensive Half-Space Actions: {metrics['offensive_half_space_actions']}")
    print("-" * 30)

    # 6. Advanced Metrics
    print("\n6. Advanced Metrics (Field Tilt & Convex Hull)...")
    try:
        from src.features_advanced import calculate_field_tilt, calculate_team_convex_hull
    except ImportError:
        from features_advanced import calculate_field_tilt, calculate_team_convex_hull
    
    field_tilt = calculate_field_tilt(events)
    hull_areas = calculate_team_convex_hull(events)
    
    print("\n--- Field Tilt (Final Third Pass Share) ---")
    for team, value in field_tilt.items():
        print(f"{team}: {value:.1%}")
        
    print("\n--- Convex Hull Area (Offensive Territory Size) ---")
    for team, area in hull_areas.items():
        # Pitch area is roughly 120*80 = 9600. Hull area tells us how spread out they played.
        print(f"{team}: {area:.2f}")

    # 7. Inspect Data
    print("\n7. Data Structure Verification:")
    print(f"DataFrame Shape: {events.shape}")
    print("\nColumns:")
    print(events.columns.tolist())
    
    print("\nFirst 5 Rows:")
    print(events.head())

if __name__ == "__main__":
    main()
