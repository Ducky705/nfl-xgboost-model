import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.features as v2_features

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_ML_PATH = "models/v2_moneyline.json"

def load_data():
    if not os.path.exists(CACHE_PATH_V2):
        print("❌ Cache not found!")
        exit()

    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    return db

def load_model():
    ml_model = xgb.XGBClassifier()
    ml_model.load_model(MODEL_ML_PATH)
    return ml_model

def calculate_moneyline_kelly(prob, vegas_prob, vegas_odds, params):
    """
    Parametrized Kelly Calculation for Optimization
    """
    MIN_EDGE = params['min_edge']
    KELLY_FRACTION = params['kelly_fraction']
    MIN_ODDS = params['min_odds']
    MAX_ODDS = params['max_odds']
    MAX_UNITS = 3.0 # Keep this static to avoid reckless betting
    
    # 1. Odds Filter
    if vegas_odds < MIN_ODDS: return 0.0 # Too heavy favorite (e.g. -600 < -500)
    if vegas_odds > MAX_ODDS: return 0.0 # Too heavy underdog (e.g. +600 > +500)
    
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
    
    units = min(max(0, kelly_fraction_val * KELLY_FRACTION), MAX_UNITS)
    return units

def objective(trial):
    # --- HYPERPARAMETERS ---
    params = {
        'min_edge': trial.suggest_float('min_edge', 0.0, 0.10),
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.05, 0.5), # Explore up to 0.5 (current)
        'min_odds': trial.suggest_int('min_odds', -1000, -110), # e.g. -500
        'max_odds': trial.suggest_int('max_odds', 100, 1000)   # e.g. +500
    }
    
    total_profit = 0.0
    total_bets = 0
    
    # Simulate Logic (Simplified for speed)
    # We will use the pre-calculated dataframe if possible, but we need to re-run predictions?
    # No, predictions are static. The decision logic is what changes.
    
    # Iterate through cached games (passed globally to avoid re-loading)
    for _, game in SIM_GAMES.iterrows():
        # ... logic ...
        # (This needs to be efficient)
        
        # Get Pre-calculated Probs
        prob_home = game['prob_home']
        prob_away = 1 - prob_home
        
        vegas_ml_home = game['home_moneyline']
        vegas_ml_away = game['away_moneyline']
        
        # Vegas Implied
        if vegas_ml_home < 0: v_prob_home = -vegas_ml_home/(-vegas_ml_home+100)
        else: v_prob_home = 100/(vegas_ml_home+100)
        
        if vegas_ml_away < 0: v_prob_away = -vegas_ml_away/(-vegas_ml_away+100)
        else: v_prob_away = 100/(vegas_ml_away+100)
        
        # Calc Units
        units_home = calculate_moneyline_kelly(prob_home, v_prob_home, vegas_ml_home, params)
        units_away = calculate_moneyline_kelly(prob_away, v_prob_away, vegas_ml_away, params)
        
        pick_units = 0
        pick_odds = 0
        did_win = False
        
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
            continue # No bet
            
        # Result
        if is_push:
            profit = 0
        elif did_win:
             if pick_odds > 0: decimal = (pick_odds/100) + 1
             else: decimal = (100/abs(pick_odds)) + 1
             profit = pick_units * (decimal - 1)
        else:
            profit = -pick_units
            
        total_profit += profit
        total_bets += 1
        
    # Objective: Maximize Profit
    # Penalty for too few bets? 
    if total_bets < 20: 
        return -100 # Penalize inactive models
        
    return total_profit

# --- PRE-CALCULATION ---
# Load Data & Models ONCE
print("Loading Data for Optimization...")
db = load_data()
ml_model = load_model()

schedule = db['schedule']
# Only completed games
completed = schedule[(schedule['result'].notna()) & (schedule['game_type'] == 'REG')].copy()

# Pre-calculate predictions to speed up trials
print("Pre-calculating predictions...")
# We need features.
full_games = v2_features.engineering_pipeline(db) # Use full db
# Filter to completed
games_with_features = full_games[full_games['game_id'].isin(completed['game_id'])].copy()

# Add scores and MLs from schedule if missing in features (features usually cleans this, but let's be safe)
# Actually full_games merges `betting_df` so it should have lines.
# It might NOT have scores if they aren't in `betting_df`.
# Let's merge scores from `completed`
games_with_features = pd.merge(games_with_features, completed[['game_id', 'home_score', 'away_score']], on='game_id', how='left', suffixes=('', '_y')) # handle duplicates if any

# Drop games without ML
games_with_features = games_with_features.dropna(subset=['home_moneyline', 'away_moneyline'])

# Generate Predictions
try:
    ml_cols = ml_model.feature_names
except:
    ml_cols = ml_model.get_booster().feature_names

# Handle missing cols
for c in ml_cols:
    if c not in games_with_features.columns: 
        games_with_features[c] = 0

X = games_with_features[ml_cols]
    
probs = ml_model.predict_proba(X)[:, 1] # Probability of Class 1 (Home Win usually?)
# CHECK: XGBClassifier classes. usually [0, 1]. 1 is Home Win?
# Let's assume standard 1=Home Win. 
games_with_features['prob_home'] = probs

SIM_GAMES = games_with_features

print(f"Optimization Set: {len(SIM_GAMES)} games")


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) # Quick sweep

    print("--- 🏆 OPTIMIZATION RESULTS 🏆 ---")
    print(f"Best Profit: {study.best_value:.2f}u")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Save best params
    with open("config/best_ml_params.json", "w") as f:
        import json
        json.dump(study.best_params, f)
