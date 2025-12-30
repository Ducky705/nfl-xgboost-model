"""
Optuna Optimization: Totals Kelly Parameters for ROI
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
MODEL_PATH = "models/v3_total_stack.pkl"
MODEL_FEATURES = "models/v3_total_features.pkl"
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

def calculate_totals_kelly(abs_edge, params):
    """Parametrized Kelly Calculation for Totals"""
    MIN_EDGE = params['min_edge']
    KELLY_FRACTION = params['kelly_fraction']
    MAX_UNITS = 2.0
    STD_DEV = 13.5
    PAYOUT_RATIO = 0.9091
    
    if abs_edge < MIN_EDGE:
        return 0.0
    
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    units = round(min(kelly_units, MAX_UNITS), 1)
    return units

def objective(trial):
    # --- HYPERPARAMETERS ---
    # Constrain search to > 3.0 edge to avoid noise.
    params = {
        'min_edge': trial.suggest_float('min_edge', 3.0, 8.0),
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.01, 0.20),
    }
    
    total_profit = 0.0
    total_wagered = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        pred_total = game['pred_total']
        raw_total = game['total_line']
        if pd.isna(raw_total): continue
        
        # Clamp pred total similar to simulate_v3.py (safety bounds)
        fair_total = max(min(pred_total, 70), 20)
        
        edge = fair_total - raw_total
        abs_edge = abs(edge)
        
        units = calculate_totals_kelly(abs_edge, params)
        
        if units > 0:
            pick_type = 'OVER' if edge > 0 else 'UNDER'
            actual_total = game['home_score'] + game['away_score']
            
            win = False
            push = False
            
            if actual_total == raw_total: push = True
            elif pick_type == 'OVER':
                if actual_total > raw_total: win = True
            else:
                if actual_total < raw_total: win = True
                
            total_wagered += units
            total_bets += 1
            
            if push:
                profit = 0.0
            elif win:
                profit = units * 0.9091
            else:
                profit = -units
            
            total_profit += profit
            
    if total_bets < MIN_BETS:
        return -1000
    
    if total_wagered > 0:
        roi = total_profit / total_wagered
    else:
        roi = -1.0
        
    # Constraint: Min ROI 7%
    if roi < 0.07:
        return -500 + (roi * 100)
        
    # Objective: Maximize Total Profit
    return total_profit

# --- PRE-CALCULATION ---
print("Loading Data for Totals ROI Optimization...")
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

games_with_features = pd.merge(
    games_with_features, 
    completed[['game_id', 'home_score', 'away_score', 'total_line']], 
    on='game_id', 
    how='left', 
    suffixes=('', '_y')
)

games_with_features = games_with_features.dropna(subset=['total_line'])

for c in features:
    if c not in games_with_features.columns: games_with_features[c] = 0

X = games_with_features[features]
games_with_features['pred_total'] = model.predict(X)

SIM_GAMES = games_with_features
print(f"Optimization Set: {len(SIM_GAMES)} games")

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    print("\n🚀 Starting Totals Optimization (200 trials)...\n")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("\n" + "="*60)
    print("🏆 TOTALS ROI OPTIMIZATION RESULTS 🏆")
    print("="*60)
    print(f"Best ROI: {study.best_value:.2f}%")
    print("\nOptimal Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Validation
    params = study.best_params
    total_profit = 0.0
    wins = 0
    losses = 0
    pushes = 0
    total_wagered = 0.0
    
    for _, game in SIM_GAMES.iterrows():
        pred_total = game['pred_total']
        raw_total = game['total_line']
        fair_total = max(min(pred_total, 70), 20)
        edge = fair_total - raw_total
        
        units = calculate_totals_kelly(abs(edge), params)
        
        if units > 0:
            pick_type = 'OVER' if edge > 0 else 'UNDER'
            actual_total = game['home_score'] + game['away_score']
            
            win = False
            push = False
            
            if actual_total == raw_total: push = True
            elif pick_type == 'OVER':
                 if actual_total > raw_total: win = True
            else:
                 if actual_total < raw_total: win = True
            
            total_wagered += units
            if push:
                pushes += 1
            elif win:
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
    with open("config/best_totals_roi_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
