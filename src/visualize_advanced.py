import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch, Pitch
from statsbombpy import sb
from scipy.spatial import ConvexHull
import traceback

os.makedirs('outputs/plots', exist_ok=True)

# Define Zone 14 and Half-Spaces coordinates (StatsBomb: 120x80)
# Zone 14: Just outside the box horizontally, central vertically
Z14_X_MIN, Z14_X_MAX = 80, 100
Z14_Y_MIN, Z14_Y_MAX = 30, 50

# Left Half-Space
LHS_Y_MIN, LHS_Y_MAX = 18, 30
# Right Half-Space
RHS_Y_MIN, RHS_Y_MAX = 50, 62

def extract_coordinates(events_df):
    """Helper to extract x, y from StatsBomb location arrays."""
    df = events_df.copy()
    valid_locs = df[df['location'].notna()]
    df.loc[valid_locs.index, 'x'] = valid_locs['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df.loc[valid_locs.index, 'y'] = valid_locs['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
    return df

def calculate_zone_percentages(df, team_name):
    team_events = df[df['team'] == team_name]
    total = len(team_events)
    if total == 0:
        return 0, 0
    
    in_z14 = team_events[
        (team_events['x'] >= Z14_X_MIN) & (team_events['x'] <= Z14_X_MAX) &
        (team_events['y'] >= Z14_Y_MIN) & (team_events['y'] <= Z14_Y_MAX)
    ]
    
    in_hs = team_events[
        (team_events['x'] >= Z14_X_MIN) & (team_events['x'] <= Z14_X_MAX) &
        ((team_events['y'] >= LHS_Y_MIN) & (team_events['y'] <= LHS_Y_MAX) |
         (team_events['y'] >= RHS_Y_MIN) & (team_events['y'] <= RHS_Y_MAX))
    ]
    
    z14_pct = (len(in_z14) / total) * 100
    hs_pct = (len(in_hs) / total) * 100
    return z14_pct, hs_pct

def plot_comparative_heatmap(match_id=3869685):
    """Generates comparative density heatmaps with Zone 14 overlay."""
    print(f"Fetching events for match {match_id} (World Cup Final 2022)...")
    events = sb.events(match_id=match_id)
    
    # Filter 'Pass' and 'Carry'
    actions = events[events['type'].isin(['Pass', 'Carry'])].copy()
    actions = extract_coordinates(actions)
    actions = actions.dropna(subset=['x', 'y'])
    
    teams = actions['team'].unique()
    if len(teams) < 2:
        print("Not enough teams found.")
        return

    # Set up Pitch
    pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
    fig, axs = pitch.grid(ncols=2, title_height=0.08, endnote_height=0.04, axis=False, figheight=10)
    fig.patch.set_facecolor('#22312b')
    
    for idx, team in enumerate([teams[0], teams[1]]):
        ax = axs['pitch'][idx]
        
        team_actions = actions[actions['team'] == team]
        x = team_actions['x'].values
        y = team_actions['y'].values
        
        # Calculate Bins logic
        bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(6, 5))
        pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b', alpha=0.8)
        
        # Overlay Zone 14
        pitch.lines(Z14_X_MIN, Z14_Y_MIN, Z14_X_MAX, Z14_Y_MIN, ax=ax, lw=2, color='cyan', alpha=0.7)
        pitch.lines(Z14_X_MIN, Z14_Y_MAX, Z14_X_MAX, Z14_Y_MAX, ax=ax, lw=2, color='cyan', alpha=0.7)
        pitch.lines(Z14_X_MIN, Z14_Y_MIN, Z14_X_MIN, Z14_Y_MAX, ax=ax, lw=2, color='cyan', alpha=0.7)
        pitch.lines(Z14_X_MAX, Z14_Y_MIN, Z14_X_MAX, Z14_Y_MAX, ax=ax, lw=2, color='cyan', alpha=0.7)
        
        # Calculate percentages
        z14_pct, hs_pct = calculate_zone_percentages(team_actions, team)
        
        ax.set_title(f"{team}\nZone 14: {z14_pct:.1f}% | Half-Spaces: {hs_pct:.1f}%", 
                     color='white', fontsize=14, pad=10)

    fig.suptitle("Comparativo Tático: Densidade de Ações e Zona 14 (Passes + Carries)\nArgentina vs França (Copa 2022)", 
                 color='white', fontsize=18)
    
    output_path = 'outputs/plots/comparative_heatmap_wc_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#22312b')
    plt.close()
    print(f"Heatmap saved to {output_path}")

def plot_convex_hull_phases(match_id=3869685):
    """Plots separate Convex Hulls for 1st Half and 2nd Half."""
    events = sb.events(match_id=match_id)
    passes = events[(events['type'] == 'Pass') & (events['pass_outcome'].isna())].copy()
    passes = extract_coordinates(passes)
    passes = passes.dropna(subset=['x', 'y'])
    
    teams = passes['team'].unique()
    periods = [1, 2] # 1st and 2nd Half
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#efefef')
    fig, axs = pitch.grid(nrows=2, ncols=2, figheight=12,
                          title_height=0.08, endnote_height=0.04, axis=False)
    fig.patch.set_facecolor('#22312b')
    
    for row_idx, period in enumerate(periods):
        for col_idx, team in enumerate([teams[0], teams[1]]):
            ax = axs['pitch'][row_idx, col_idx]
            
            period_passes = passes[(passes['team'] == team) & (passes['period'] == period)]
            
            # Find average position of starting 11 players
            player_avg = period_passes.groupby('player').agg({'x': 'mean', 'y': 'mean', 'id': 'count'}).reset_index()
            # Filter players who played significantly (e.g., at least 5 passes in the half)
            starters = player_avg[player_avg['id'] >= 5]
            
            pitch.scatter(starters['x'], starters['y'], s=100, color='white', edgecolors='black', ax=ax, zorder=3)
            
            if len(starters) >= 3:
                points = starters[['x', 'y']].values
                hull = ConvexHull(points)
                # Draw the hull
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], '-', color='gold', lw=2, zorder=2)
                # Fill hull
                hull_points = points[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], color='gold', alpha=0.3, zorder=1)
                
                # Calculate Area
                hull_area = hull.volume
            else:
                hull_area = 0
            
            period_name = "1º Tempo" if period == 1 else "2º Tempo"
            ax.set_title(f"{team} ({period_name})\nHull Area: {hull_area:.1f}", color='white', fontsize=12)
            
    fig.suptitle("Evolução Territorial: Convex Hulls (Média de Posições)\nArgentina vs França (Copa 2022)", 
                 color='white', fontsize=18)
                 
    output_path = 'outputs/plots/convex_hull_phases_wc_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#22312b')
    plt.close()
    print(f"Convex Hull comparison saved to {output_path}")

if __name__ == "__main__":
    print("--- Starting Phase 2: Advanced Visualization ---")
    try:
        plot_comparative_heatmap(match_id=3869685)
        plot_convex_hull_phases(match_id=3869685)
        print("Visualizations successfully generated in outputs/plots/")
    except Exception as e:
        print(f"Error during visualization generation: {e}")
        traceback.print_exc()
