"""
Optuna Optimization: Moneyline Kelly Parameters for ROI
Objective: Maximize ROI with a MINIMUM of 15 bets in the 2025 season.
"""
import optuna
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.features as v2_features

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_ML_PATH = "models/v3_moneyline_stack.pkl"
SEASON_FILTER = 2025
MIN_BETS = 15

def load_data():
    if not os.path.exists(CACHE_PATH_V2):
        print("❌ Cache not found!")
        exit()

    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    return db

def load_model():
    with open(MODEL_ML_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_moneyline_kelly(prob, vegas_prob, vegas_odds, params):
    """Parametrized Kelly Calculation for Optimization"""
    MIN_EDGE = params['min_edge']
    KELLY_FRACTION = params['kelly_fraction']
    MIN_ODDS = params['min_odds']
    MAX_ODDS = params['max_odds']
    MAX_UNITS = 2.0
    
    # 1. Odds Filter
    if vegas_odds < MIN_ODDS: return 0.0
    if vegas_odds > MAX_ODDS: return 0.0
    
    # 2. Edge Filter
    edge = prob - vegas_prob
    if edge < MIN_EDGE:
        return 0.0
    
    # 3. Kelly Calculation
    if vegas_odds < 0:
        payout = 100 / abs(vegas_odds)
    else:
        payout = vegas_odds / 100
        
    b = payout
    p = prob
    q = 1 - p
    
    kelly_fraction_val = (b * p - q) / b if b > 0 else 0
    
    units = round(min(max(0, kelly_fraction_val * KELLY_FRACTION), MAX_UNITS), 1)
    return units

def objective(trial):
    # --- HYPERPARAMETERS ---
    params = {
        'min_edge': trial.suggest_float('min_edge', 0.05, 0.20),  # 5% to 20% edge
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.5, 3.0),
        'min_odds': trial.suggest_int('min_odds', -600, -120),
        'max_odds': trial.suggest_int('max_odds', 120, 500)
    }
    
    total_profit = 0.0
    total_wagered = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        prob_home = game['prob_home']
        prob_away = 1 - prob_home
        
        vegas_ml_home = game['home_moneyline']
        vegas_ml_away = game['away_moneyline']
        
        # Vegas Implied Probabilities
        if vegas_ml_home < 0: 
            v_prob_home = -vegas_ml_home / (-vegas_ml_home + 100)
        else: 
            v_prob_home = 100 / (vegas_ml_home + 100)
        
        if vegas_ml_away < 0: 
            v_prob_away = -vegas_ml_away / (-vegas_ml_away + 100)
        else: 
            v_prob_away = 100 / (vegas_ml_away + 100)
        
        # Calculate Units
        units_home = calculate_moneyline_kelly(prob_home, v_prob_home, vegas_ml_home, params)
        units_away = calculate_moneyline_kelly(prob_away, v_prob_away, vegas_ml_away, params)
        
        pick_units = 0
        pick_odds = 0
        did_win = False
        is_push = False
        
        if units_home > 0 and units_home >= units_away:
            pick_units = units_home
            pick_odds = vegas_ml_home
            did_win = game['home_score'] > game['away_score']
            is_push = game['home_score'] == game['away_score']
        elif units_away > 0:
            pick_units = units_away
            pick_odds = vegas_ml_away
            did_win = game['away_score'] > game['home_score']
            is_push = game['home_score'] == game['away_score']
        else:
            continue
            
        total_wagered += pick_units
        
        if is_push:
            profit = 0
        elif did_win:
            if pick_odds > 0: 
                decimal = (pick_odds / 100) + 1
            else: 
                decimal = (100 / abs(pick_odds)) + 1
            profit = pick_units * (decimal - 1)
        else:
            profit = -pick_units
            
        total_profit += profit
        total_bets += 1
    
    # Constraint: Min 20 Bets (User wants volume)
    if total_bets < 20: 
        return -1000 
    
    # Calculate ROI
    if total_wagered > 0:
        roi = total_profit / total_wagered
    else:
        roi = -1.0
        
    # Constraint: Min ROI of 5% (Safety floor)
    if roi < 0.05:
        return -500 + (roi * 100)
        
    # Objective: Maximize Total Profit
    # This aligns with user's specific request for more volume + profit over pure ROI
    return total_profit

# --- PRE-CALCULATION ---
print("Loading Data for Moneyline ROI Optimization...")
db = load_data()
ml_model = load_model()

schedule = db['schedule']
# Only completed games in 2025 Regular Season
completed = schedule[
    (schedule['result'].notna()) & 
    (schedule['game_type'] == 'REG') &
    (schedule['season'] == SEASON_FILTER)
].copy()

print(f"Found {len(completed)} completed 2025 games")

# Pre-calculate predictions
print("Running Feature Engineering Pipeline...")
full_games = v2_features.engineering_pipeline(db)
games_with_features = full_games[full_games['game_id'].isin(completed['game_id'])].copy()

# Merge scores
games_with_features = pd.merge(
    games_with_features, 
    completed[['game_id', 'home_score', 'away_score']], 
    on='game_id', 
    how='left', 
    suffixes=('', '_y')
)

# Drop games without ML odds
games_with_features = games_with_features.dropna(subset=['home_moneyline', 'away_moneyline'])

# Generate Predictions
with open("models/v3_moneyline_features.pkl", 'rb') as f:
    ml_cols = pickle.load(f)

for c in ml_cols:
    if c not in games_with_features.columns: 
        games_with_features[c] = 0

X = games_with_features[ml_cols]
probs = ml_model.predict_proba(X)[:, 1]
games_with_features['prob_home'] = probs

SIM_GAMES = games_with_features

print(f"Optimization Set: {len(SIM_GAMES)} games with ML odds")


if __name__ == "__main__":
    # Use TPE sampler for better exploration
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    print("\n🚀 Starting Optuna Optimization (200 trials)...\n")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("\n" + "="*60)
    print("🏆 MONEYLINE ROI OPTIMIZATION RESULTS 🏆")
    print("="*60)
    print(f"Best ROI: {study.best_value:.2f}%")
    print("\nOptimal Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Run once more with best params to show stats
    params = study.best_params
    total_profit = 0.0
    total_wagered = 0.0
    wins = 0
    losses = 0
    
    for _, game in SIM_GAMES.iterrows():
        prob_home = game['prob_home']
        prob_away = 1 - prob_home
        vegas_ml_home = game['home_moneyline']
        vegas_ml_away = game['away_moneyline']
        
        if vegas_ml_home < 0: 
            v_prob_home = -vegas_ml_home / (-vegas_ml_home + 100)
        else: 
            v_prob_home = 100 / (vegas_ml_home + 100)
        if vegas_ml_away < 0: 
            v_prob_away = -vegas_ml_away / (-vegas_ml_away + 100)
        else: 
            v_prob_away = 100 / (vegas_ml_away + 100)
        
        units_home = calculate_moneyline_kelly(prob_home, v_prob_home, vegas_ml_home, params)
        units_away = calculate_moneyline_kelly(prob_away, v_prob_away, vegas_ml_away, params)
        
        if units_home > 0 and units_home >= units_away:
            pick_units = units_home
            pick_odds = vegas_ml_home
            did_win = game['home_score'] > game['away_score']
        elif units_away > 0:
            pick_units = units_away
            pick_odds = vegas_ml_away
            did_win = game['away_score'] > game['home_score']
        else:
            continue
            
        total_wagered += pick_units
        
        if did_win:
            if pick_odds > 0: 
                decimal = (pick_odds / 100) + 1
            else: 
                decimal = (100 / abs(pick_odds)) + 1
            profit = pick_units * (decimal - 1)
            wins += 1
        else:
            profit = -pick_units
            losses += 1
            
        total_profit += profit
    
    print(f"\n📊 Backtest Results with Optimal Params:")
    print(f"  Record: {wins}-{losses}-0")
    print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
    print(f"  Total Profit: {total_profit:.2f}u")
    print(f"  Total Wagered: {total_wagered:.2f}u")
    print(f"  ROI: {(total_profit/total_wagered)*100:.2f}%")
    
    # Save best params
    import json
    with open("config/best_ml_roi_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n✅ Saved optimal params to best_ml_roi_params.json")
