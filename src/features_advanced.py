import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

def calculate_field_tilt(events_df):
    """
    Calculates Field Tilt for each team.
    Field Tilt = Share of passes in the final third (x >= 80).
    
    Args:
        events_df (pd.DataFrame): Match events.
        
    Returns:
        dict: {team_name: field_tilt_percentage}
    """
    df = events_df.copy()
    
    # Filter for Passes
    passes = df[df['type'] == 'Pass'].copy()
    
    # Extract x coordinate
    passes['x'] = passes['location'].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) else 0)
    
    # Filter Final Third
    final_third_passes = passes[passes['x'] >= 80]
    
    # Count per team
    team_counts = final_third_passes['team'].value_counts()
    total_final_third = team_counts.sum()
    
    if total_final_third == 0:
        return {team: 0.0 for team in df['team'].unique()}
        
    field_tilt = (team_counts / total_final_third).to_dict()
    
    # Ensure all teams are present even if 0 passes
    for team in df['team'].unique():
        if team not in field_tilt:
            field_tilt[team] = 0.0
            
    return field_tilt

def calculate_team_convex_hull(events_df):
    """
    Calculates the area of the Convex Hull formed by completed passes for each team.
    
    Args:
        events_df (pd.DataFrame): Match events.
        
    Returns:
        dict: {team_name: hull_area}
    """
    df = events_df.copy()
    
    # Filter Completed Passes
    # In StatsBomb, if pass_outcome is NaN/Null, it is complete.
    completed_passes = df[
        (df['type'] == 'Pass') & 
        (df['pass_outcome'].isna())
    ].copy()
    
    hull_areas = {}
    
    teams = completed_passes['team'].unique()
    
    for team in teams:
        team_passes = completed_passes[completed_passes['team'] == team]
        
        # specific check for location validity
        points = team_passes['location'].apply(lambda x: x if isinstance(x, (list, tuple)) else np.nan).dropna().tolist()
        
        if len(points) < 3:
            hull_areas[team] = 0.0
            continue
            
        points = np.array(points)
        
        try:
            # For 2D, volume is the Area. area is the Perimeter.
            hull = ConvexHull(points)
            hull_areas[team] = hull.volume 
        except Exception as e:
            # Fallback for collinear points or other errors
            hull_areas[team] = 0.0
            
    return hull_areas
