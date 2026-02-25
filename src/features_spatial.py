import pandas as pd
import numpy as np

def assign_zone_18(x, y):
    """
    Assigns a zone number (1-18) based on x, y coordinates on a 120x80 pitch.
    
    Grid: 6 columns x 3 rows.
    
    Calculation:
    - X bins: 0-20, 20-40, 40-60, 60-80, 80-100, 100-120 (Indices 0-5)
    - Y bins: 0-26.67, 26.67-53.33, 53.33-80 (Indices 0-2)
    
    Formula: Zone = (Col_Index * 3) + Row_Index + 1
    
    Zone 14 Logic:
    - X in [80, 100] -> Col Index 4
    - Y in [26.67, 53.33] -> Row Index 1
    - Zone = 4*3 + 1 + 1 = 14
    """
    # Clip coordinates to pitch dimensions just in case
    x = max(0, min(120, x))
    y = max(0, min(80, y))
    
    # Calculate column index (0-5)
    # Using 120.01 to handle x=120 usually falling into next bin if not careful, 
    # but floor(120/20) = 6 which is out of bounds 0-5.
    if x == 120:
        col = 5
    else:
        col = int(x / 20)
        
    # Calculate row index (0-2)
    if y == 80:
        row = 2
    else:
        row = int(y / (80/3))
        
    return (col * 3) + row + 1

def calculate_zone_metrics(events_df):
    """
    Calculates spatial metrics for the match.
    
    Args:
        events_df (pd.DataFrame): DataFrame containing match events with 'location' column.
        
    Returns:
        dict: calculated metrics.
    """
    # Ensure we use a copy to avoid SettingWithCopyWarning on the original df if it's a slice
    df = events_df.copy()
    
    # Extract x and y from location
    # location is a list [x, y] or NaN
    # We drop Generic events with no location for spatial analysis
    spatial_data = df.dropna(subset=['location']).copy()
    
    spatial_data['x'] = spatial_data['location'].apply(lambda loc: loc[0])
    spatial_data['y'] = spatial_data['location'].apply(lambda loc: loc[1])
    
    # Assign Zones
    spatial_data['zone'] = spatial_data.apply(lambda row: assign_zone_18(row['x'], row['y']), axis=1)
    
    # --- Metrics Calculation ---
    
    # 1. Zone 14 Passes
    # Zone 14 is defined by our assign_zone_18 function (see docstring)
    zone_14_passes = spatial_data[
        (spatial_data['zone'] == 14) & 
        (spatial_data['type'] == 'Pass')
    ]
    
    # 2. Shots from inside the box
    # Box dimensions usually: X >= 102, Y roughly between 18 and 62
    box_shots = spatial_data[
        (spatial_data['type'] == 'Shot') &
        (spatial_data['x'] >= 102) &
        (spatial_data['y'] >= 18) & 
        (spatial_data['y'] <= 62)
    ]
    
    # 3. Offensive Half-Spaces
    # Defining Half-spaces roughly as the vertical channels between center and wing.
    # Center is Y [30, 50], Wings are [0, 15] and [65, 80]... 
    # Let's use specific definitions often cited in analytics:
    # "Half-spaces" are typically Y in [15-30] and [50-65] approximate.
    # "Offensive" -> X > 60 (attacking half)
    half_space_mask = (
        (spatial_data['x'] > 60) & 
        (
            ((spatial_data['y'] >= 18) & (spatial_data['y'] <= 32)) | # Left-ish HS
            ((spatial_data['y'] >= 48) & (spatial_data['y'] <= 62))   # Right-ish HS
        )
    )
    half_space_actions = spatial_data[half_space_mask]
    
    return {
        "zone_14_passes": len(zone_14_passes),
        "box_shots": len(box_shots),
        "offensive_half_space_actions": len(half_space_actions)
    }
