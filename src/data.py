
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import os
import pickle
from datetime import datetime

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
TEAM_MAP = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'SD': 'LAC', 'SL': 'LA', 'STL': 'LA', 'OAK': 'LV'}
POSITIONS_OF_INTEREST = ['QB', 'RB', 'WR', 'TE', 'OL', 'T', 'G', 'C', 'DL', 'DT', 'DE', 'LB', 'CB', 'S', 'DB']

now = datetime.now()
CURRENT_SEASON = now.year if now.month > 2 else now.year - 1
YEARS = list(range(2018, CURRENT_SEASON + 1))

def process_injuries(years):
    print("Fetching Injury Data...")
    try:
        injuries = nfl.import_injuries(years)
        # Filter for relevant positions and statuses
        injuries = injuries[injuries['position'].isin(POSITIONS_OF_INTEREST)]
        
        # Normalize status
        # report_status: 'Out', 'Doubtful', 'Questionable'
        # We care most about confirmed miss or likely miss
        injuries['is_likely_out'] = injuries['report_status'].isin(['Out', 'Doubtful']).astype(int)
        
        # Group by team and week to get counts
        # We need to map 'team' column. nfl_data_py usually uses 'team' or 'club_code'
        injuries['team'] = injuries['team'].replace(TEAM_MAP)
        
        injury_counts = injuries[injuries['is_likely_out'] == 1].groupby(['season', 'week', 'team']).agg({
            'is_likely_out': 'count',
            'position': lambda x: list(x)
        }).reset_index().rename(columns={'is_likely_out': 'starters_out_count'})
        
        # Specific QB check
        qb_injuries = injuries[(injuries['position'] == 'QB') & (injuries['is_likely_out'] == 1)]
        qb_out = qb_injuries.groupby(['season', 'week', 'team']).size().reset_index(name='qb_out_count')
        
        injury_features = injury_counts.merge(qb_out, on=['season', 'week', 'team'], how='left').fillna(0)
        injury_features['qb_starter_out'] = (injury_features['qb_out_count'] > 0).astype(int)
        
        return injury_features
    except Exception as e:
        print(f"Error processing injuries: {e}")
        return pd.DataFrame()

def get_base_stats(df, full_pbp):
    print("Aggregating Base Stats...")
    # 1. General Offense
    gen = df.groupby(['season', 'week', 'posteam']).agg({
        'epa': 'mean', 
        'yards_gained': 'mean',
        'wp': 'mean' # Win probability at start of play can be proxy for dominance? No, average WP isn't great.
    }).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa', 'yards_gained': 'off_ypp'})

    # 1b. General Defense
    gen_def = df.groupby(['season', 'week', 'defteam']).agg({
        'epa': 'mean', 
        'yards_gained': 'mean'
    }).reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa', 'yards_gained': 'def_ypp'})

    # 2. Early Down Success Rate (Standard)
    edsr = df[df['down'].isin([1, 2])].groupby(['season', 'week', 'posteam'])['success'].mean().reset_index().rename(columns={'posteam': 'team', 'success': 'off_edsr'})
    def_edsr = df[df['down'].isin([1, 2])].groupby(['season', 'week', 'defteam'])['success'].mean().reset_index().rename(columns={'defteam': 'team', 'success': 'def_edsr'})

    # 3. Passing / Rushing Splits (Offense)
    pass_df = df[df['pass'] == 1].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_pass_epa'})
    rush_df = df[df['rush'] == 1].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_rush_epa'})

    # 3b. Passing / Rushing Splits (Defense)
    def_pass_df = df[df['pass'] == 1].groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index().rename(columns={'defteam': 'team', 'epa': 'def_pass_epa'})
    def_rush_df = df[df['rush'] == 1].groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index().rename(columns={'defteam': 'team', 'epa': 'def_rush_epa'})

    # 4. Sacks (Offensive allowed & Defensive generated)
    pass_plays = full_pbp[full_pbp['pass'] == 1]
    off_sacks = pass_plays.groupby(['season', 'week', 'posteam'])['sack'].mean().reset_index().rename(columns={'posteam': 'team', 'sack': 'off_sack_rate'})
    def_sacks = pass_plays.groupby(['season', 'week', 'defteam'])['sack'].mean().reset_index().rename(columns={'defteam': 'team', 'sack': 'def_sack_rate'})

    # 5. Turnovers
    tos = full_pbp.groupby(['season', 'week', 'posteam']).agg({'fumble_lost': 'sum', 'interception': 'sum'}).reset_index()
    tos['turnovers_lost'] = tos['fumble_lost'] + tos['interception']
    def_tos = full_pbp.groupby(['season', 'week', 'defteam']).agg({'fumble_lost': 'sum', 'interception': 'sum'}).reset_index()
    def_tos['turnovers_gained'] = def_tos['fumble_lost'] + def_tos['interception']

    # 6. Red Zone
    rz = df[df['yardline_100'] <= 20].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_rz_epa'})
    
    # 7. Merge All
    merged = gen.merge(edsr, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(pass_df, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(rush_df, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(off_sacks, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(def_sacks, on=['season', 'week', 'team'], how='left')
    # Merge Defense
    merged = merged.merge(gen_def, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(def_edsr, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(def_pass_df, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(def_rush_df, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(tos[['season', 'week', 'posteam', 'turnovers_lost']].rename(columns={'posteam': 'team'}), on=['season', 'week', 'team'], how='left')
    merged = merged.merge(def_tos[['season', 'week', 'defteam', 'turnovers_gained']].rename(columns={'defteam': 'team'}), on=['season', 'week', 'team'], how='left')
    merged = merged.merge(rz, on=['season', 'week', 'team'], how='left')
    
    return merged.fillna(0)

def get_historic_starters(pbp):
    """
    Identifies the QB starter for each game based on majority of dropbacks.
    Returns DataFrame: [season, week, team, qb_name, qb_id]
    """
    print("Extracting Historical QB Starters...")
    
    # Filter for Pass Plays or Sacks (Dropbacks)
    dropbacks = pbp[
        (pbp['play_type'].isin(['pass', 'run'])) & 
        (pbp['qb_dropback'] == 1) &
        (~pbp['name'].isna())
    ].copy()
    
    # Count dropbacks per QB per game
    qb_counts = dropbacks.groupby(['season', 'week', 'posteam', 'name', 'passer_id']).size().reset_index(name='dropbacks')
    
    # Sort by dropbacks descending
    qb_counts = qb_counts.sort_values(['season', 'week', 'posteam', 'dropbacks'], ascending=[True, True, True, False])
    
    # Take top QB per game
    starters = qb_counts.drop_duplicates(subset=['season', 'week', 'posteam'], keep='first')
    
    # Rename for consistency
    starters = starters.rename(columns={'posteam': 'team', 'name': 'qb_name', 'passer_id': 'qb_id'})
    
    return starters[['season', 'week', 'team', 'qb_name', 'qb_id']]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print(f"Downloading Data ({min(YEARS)}-{max(YEARS)})...")
    schedule = nfl.import_schedules(YEARS)
    pbp = nfl.import_pbp_data(YEARS)
    
    print("Cleaning Data...")
    pbp['posteam'] = pbp['posteam'].replace(TEAM_MAP)
    pbp['defteam'] = pbp['defteam'].replace(TEAM_MAP)
    
    # Filter for standard plays
    pbp_clean = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & (pbp['wp'] > 0.05) & (pbp['wp'] < 0.95)].dropna(subset=['epa', 'posteam', 'defteam', 'success', 'yards_gained'])
    
    # Get Stats
    base_stats = get_base_stats(pbp_clean, pbp)
    injury_stats = process_injuries(YEARS)
    qb_starters = get_historic_starters(pbp)
    
    # Save Raw Data Components
    raw_data = {
        'schedule': schedule,
        'base_stats': base_stats,
        'injury_stats': injury_stats,
        'qb_starters': qb_starters,
        'pbp_clean': pbp_clean[['season', 'week', 'posteam', 'defteam', 'epa', 'play_type', 'passer_id', 'name']], # Keep lightweight but carry QB identifiers
        'timestamp': datetime.now()
    }
    
    with open(CACHE_PATH_V2, 'wb') as f:
        pickle.dump(raw_data, f)
        
    print(f"Data Ingestion Complete. Saved to {CACHE_PATH_V2}")
