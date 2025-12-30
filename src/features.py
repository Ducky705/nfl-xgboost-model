
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering.team_stats_generator import TeamStatsGenerator
from src.feature_engineering.qb_stats_generator import QBStatsGenerator
from src.feature_engineering.situational_generator import SituationalGenerator
from src.feature_engineering.market_generator import MarketGenerator

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"

def engineering_pipeline(raw_data):
    print("Starting v3.0 Feature Engineering Pipeline...")
    
    # 1. Team Stats (Base + Betting + Rolling)
    team_gen = TeamStatsGenerator(raw_data)
    rolling_features = team_gen.generate()
    
    # 2. QB Stats (Individual EPA history)
    qb_gen = QBStatsGenerator(raw_data)
    qb_features = qb_gen.generate()
    
    # 3. Situational (Rest, Travel, Primetime)
    sit_gen = SituationalGenerator(raw_data)
    sit_features = sit_gen.generate()
    
    # 4. Market (Vegas Expectations)
    mark_gen = MarketGenerator(raw_data)
    mark_features = mark_gen.generate()
    
    print("   Merging Feature Sets...")
    
    # Base is Situational (Game Level)
    games = sit_features.copy()
    
    # Merge Market
    games = games.merge(mark_features, on=['season', 'week', 'home_team', 'away_team'], how='left')
    
    # Merge Team Stats (Home & Away)
    # rolling_features has [season, week, team, stats...]
    for side in ['home', 'away']:
        games = games.merge(rolling_features, left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
        
        # Rename columns to home_xxx / away_xxx
        rename_cols = {c: f'{side}_{c}' for c in rolling_features.columns if c not in ['season', 'week', 'team']}
        games = games.rename(columns=rename_cols)
        games = games.drop(columns=['team'])

    # Merge QB Stats
    if not qb_features.empty:
        for side in ['home', 'away']:
            # We need to know WHICH qb played for that team in that game to lookup their stats
            # In v4, we use the QB who STARTING that game.
            # qb_features has stats calculated from PRIOR games.
            # But wait, qb_features is indexed by [season, week, team].
            # AND it represents the ENTERING stats of the starter.
            # So we can just merge on team-week.
            
            # Note: QBStatsGenerator returns [season, week, team, qb_epa_L3, etc]
            # This row represents the starter for that Week's stats history.
            
            games = games.merge(qb_features, left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
            rename_cols = {c: f'{side}_{c}' for c in qb_features.columns if c not in ['season', 'week', 'team']}
            games = games.rename(columns=rename_cols)
            games = games.drop(columns=['team'])
            
    # 4. Differentials & Interaction Terms
    print("   Calculating Differentials...")
    
    # Team Diffs
    metrics = [
        'off_epa', 'off_ypp', 'off_pass_epa', 'off_rush_epa', 
        'off_edsr', 'off_sack_rate', 'def_sack_rate', 'off_rz_epa',
        'def_epa', 'def_pass_epa'
    ]
    
    for m in metrics:
        for w in ['L3', 'L5', 'L10', 'season']:
            col = f"{m}_{w}"
            if f"home_{col}" in games.columns and f"away_{col}" in games.columns:
                games[f"diff_{col}"] = games[f"home_{col}"] - games[f"away_{col}"]
    
    # QB Diffs
    if 'home_qb_epa_L3' in games.columns:
        games['qb_epa_diff_L3'] = games['home_qb_epa_L3'] - games['away_qb_epa_L3']
        games['qb_epa_diff_L10'] = games['home_qb_epa_L10'] - games['away_qb_epa_L10']
        games['qb_epa_diff_career'] = games['home_qb_epa_career'] - games['away_qb_epa_career']
        
    # Market Diffs (Vegas vs Reality)
    # Are the teams scoring more or less than Vegas implies?
    if 'implied_home_score' in games.columns and 'home_points_scored_L5' in games.columns:
        games['home_market_performance'] = games['home_points_scored_L5'] - games['implied_home_score']
        games['away_market_performance'] = games['away_points_scored_L5'] - games['implied_away_score']
        games['market_guardrail'] = games['home_market_performance'] - games['away_market_performance']
        
    # Situational Diffs were done in Generator, but we can double check
    # (Rest Diff is already there)
    
    # 5. Fill NaNs
    print("   Filling NaNs...")
    num_cols = games.select_dtypes(include=[np.number]).columns
    cols_to_fill = [c for c in num_cols if c not in ['is_primetime', 'is_division_game']] # Labels usually safe
    games[cols_to_fill] = games[cols_to_fill].fillna(0)
    
    # Add Result (Target)
    # We need scores to calculate result
    if 'home_score' not in games.columns:
        # Map them from schedule in raw_data
        sched = raw_data['schedule']
        scores = sched[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline', 'game_type']]
        scores = scores[scores['game_type'] == 'REG']
        
        # Merge scores and betting lines
        games = games.merge(scores[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'spread_line', 'total_line', 'home_moneyline', 'away_moneyline']], 
                            on=['season', 'week', 'home_team', 'away_team'], how='left')
                            
    games['result'] = games['home_score'] - games['away_score']
    
    # Drop Week 18s? No, keep all.
    
    return games

if __name__ == "__main__":
    print(f"Loading Raw Data: {CACHE_PATH_V2}")
    if not os.path.exists(CACHE_PATH_V2):
        print(f"❌ File not found: {CACHE_PATH_V2}")
        print("Run 'python src/data.py' first.")
        exit(1)
        
    with open(CACHE_PATH_V2, 'rb') as f:
        data = pickle.load(f)
        
    try:
        full_games = engineering_pipeline(data)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        exit(1)
    
    print(f"Saving Features: {FEATURES_PATH_V2}")
    print(f"   Shape: {full_games.shape}")
    print(f"   Columns: {len(full_games.columns)}")
    
    with open(FEATURES_PATH_V2, 'wb') as f:
        pickle.dump(full_games, f)
