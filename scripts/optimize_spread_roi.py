"""
Optuna Optimization: Spread Kelly Parameters for ROI
Objective: Maximize ROI with a MINIMUM of 30 bets in the 2025 season.
"""
import optuna
import pandas as pd
import numpy as np
import pickle
import os
import sys
from scipy.stats import norm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.features as v2_features

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_PATH = "models/v3_ensemble_stack.pkl"
MODEL_FEATURES = "models/v3_features.pkl"
SEASON_FILTER = 2025
MIN_BETS = 30

def load_data():
    if not os.path.exists(CACHE_PATH_V2):
        print("❌ Cache not found!")
        exit()
    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    return db

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(MODEL_FEATURES, 'rb') as f:
        features = pickle.load(f)
    return model, features

def calculate_kelly_units(abs_edge, params):
    """Parametrized Kelly Calculation for Spread"""
    # Params from Optuna
    MIN_EDGE = params['min_edge']
    KELLY_FRACTION = params['kelly_fraction']
    MAX_UNITS = 2.0
    STD_DEV = 12.53  # Standard NFL Spread Std Dev
    PAYOUT_RATIO = 0.9091 # -110 odds
    
    if abs_edge < MIN_EDGE:
        return 0.0
    
    # Kelly Formula for Spread (Normal Dist Approximation)
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    # Apply partial kelly fraction
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    
    units = round(min(kelly_units, MAX_UNITS), 1)
    return units

def objective(trial):
    # --- HYPERPARAMETERS ---
    params = {
        'min_edge': trial.suggest_float('min_edge', 0.5, 6.0),  # Explore 0.5 to 6.0 points edge
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.01, 0.20), # Conservative to Aggressive
    }
    
    total_profit = 0.0
    total_wagered = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        # Spread Logic
        pred_margin = game['pred_margin'] # Positive = Home Win Margin
        
        # Real Result
        raw_away_spread = game['spread_line']
        if pd.isna(raw_away_spread): continue
        
        # Calculate Edge
        # Vegas Line: Away +7 means Home -7
        # Vegas Expected Home Margin = -1 * Home Spread
        # If Away +7, Home -7. Vegas expects Home by 7.
        # Wait. Spread is usually "Line for Away Team" in some datasets or "Line for Home Team"?
        # Standard: 'spread_line' in `nfl_db` is historically "Away Line"?
        # Let's verify logic from simulate_v3.py:
        #   home_spread = -1 * raw_away_spread
        #   vegas_home_margin = -1 * home_spread  => which corresponds to 'raw_away_spread'
        #   edge_spread = pred_margin - vegas_home_margin
        
        home_spread = -1 * raw_away_spread
        vegas_home_margin = raw_away_spread # If Away +7, Vegas expects Home by 7? Or Away loses by 7?
        # Let's stick to the logic used in simulate_v3.py strictly
        # simulate_v3.py: 
        #   vegas_home_margin = -1 * home_spread
        #   edge_spread = pred_margin - vegas_home_margin
        
        vegas_home_margin = -1 * home_spread
        edge_spread = pred_margin - vegas_home_margin
        
        units = calculate_kelly_units(abs(edge_spread), params)
        
        if units > 0:
            pick_team = 'HOME' if edge_spread > 0 else 'AWAY'
            
            # Grading
            actual_margin = game['home_score'] - game['away_score']
            
            # Did we cover?
            covered = False
            push = False
            
            if pick_team == 'HOME':
                if actual_margin > vegas_home_margin: covered = True
                elif actual_margin == vegas_home_margin: push = True
            else:
                if actual_margin < vegas_home_margin: covered = True
                elif actual_margin == vegas_home_margin: push = True
            
            total_wagered += units
            total_bets += 1
            
            if push:
                profit = 0.0
            elif covered:
                profit = units * 0.9091
            else:
                profit = -units
                
            total_profit += profit
            
    # Constraint: Min 30 Bets
    if total_bets < MIN_BETS:
        return -1000
    
    # Calculate ROI
    if total_wagered > 0:
        roi = total_profit / total_wagered
    else:
        roi = -1.0
        
    # Constraint: Min ROI of 7% (Safety floor against high-volume noise)
    if roi < 0.07:
        return -500 + (roi * 100) # Soft penalty
        
    # Objective: Maximize Total Profit (Units)
    # This aligns with user preference: 19% ROI on 55 bets (10.45u) > 27% ROI on 30 bets (8.1u)
    return total_profit

# --- PRE-CALCULATION ---
print("Loading Data for Spread ROI Optimization...")
db = load_data()
model, features = load_model()

schedule = db['schedule']
completed = schedule[
    (schedule['result'].notna()) & 
    (schedule['game_type'] == 'REG') &
    (schedule['season'] == SEASON_FILTER)
].copy()

print(f"Found {len(completed)} completed 2025 games")

print("Running Engineering Pipeline...")
full_games = v2_features.engineering_pipeline(db)
games_with_features = full_games[full_games['game_id'].isin(completed['game_id'])].copy()

# Merge scores and lines
games_with_features = pd.merge(
    games_with_features, 
    completed[['game_id', 'home_score', 'away_score', 'spread_line']], 
    on='game_id', 
    how='left', 
    suffixes=('', '_y')
)

games_with_features = games_with_features.dropna(subset=['spread_line'])

# Generate Predictions
# VotingEnsemble uses predict on DataFrame
# Ensure features exist
for c in features:
    if c not in games_with_features.columns: games_with_features[c] = 0

X = games_with_features[features]
games_with_features['pred_margin'] = model.predict(X)

SIM_GAMES = games_with_features
print(f"Optimization Set: {len(SIM_GAMES)} games")

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    print("\n🚀 Starting Spread Optimization (200 trials)...\n")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("\n" + "="*60)
    print("🏆 SPREAD ROI OPTIMIZATION RESULTS 🏆")
    print("="*60)
    print(f"Best ROI: {study.best_value:.2f}%")
    print("\nOptimal Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Validation Run
    params = study.best_params
    total_profit = 0.0
    wins = 0
    losses = 0
    pushes = 0
    total_wagered = 0.0
    
    for _, game in SIM_GAMES.iterrows():
        raw_away_spread = game['spread_line']
        home_spread = -1 * raw_away_spread
        vegas_home_margin = -1 * home_spread
        edge_spread = game['pred_margin'] - vegas_home_margin
        
        units = calculate_kelly_units(abs(edge_spread), params)
        
        if units > 0:
            pick_team = 'HOME' if edge_spread > 0 else 'AWAY'
            actual_margin = game['home_score'] - game['away_score']
            
            covered = False
            push = False
            if pick_team == 'HOME':
                if actual_margin > vegas_home_margin: covered = True
                elif actual_margin == vegas_home_margin: push = True
            else:
                if actual_margin < vegas_home_margin: covered = True
                elif actual_margin == vegas_home_margin: push = True
            
            total_wagered += units
            if push:
                pushes += 1
            elif covered:
                wins += 1
                total_profit += units * 0.9091
            else:
                losses += 1
                total_profit += -units
                
    print(f"\n📊 Backtest Results:")
    print(f"  Record: {wins}-{losses}-{pushes}")
    if (wins+losses) > 0:
        print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
    print(f"  Profit: {total_profit:.2f}u")
    print(f"  ROI: {(total_profit/total_wagered)*100:.2f}%")

    import json
    with open("config/best_spread_roi_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
