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
MODEL_ML_PATH = "models/v4_moneyline_stack.pkl"
MODEL_ML_FEATURES = "models/v4_moneyline_features.pkl"

# Minimum bets constraint
MIN_BETS = 15

def calculate_moneyline_kelly(prob, vegas_prob, vegas_odds, params):
    MIN_EDGE = params['min_edge']
    KELLY_FRACTION = params['kelly_fraction']
    MIN_ODDS = params['min_odds_limit']
    MAX_ODDS = params['max_odds_limit']
    MAX_UNITS = 3.0
    
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
    units = min(max(0, kelly_fraction_val * KELLY_FRACTION), MAX_UNITS)
    return units

def objective(trial):
    # --- HYPERPARAMETERS ---
    params = {
        'min_edge': trial.suggest_float('min_edge', 0.05, 0.25),  # Higher minimum edge
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 1.0),
        'min_odds_limit': trial.suggest_int('min_odds_limit', -600, -150),
        'max_odds_limit': trial.suggest_int('max_odds_limit', 100, 400)
    }
    
    total_profit = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        prob_home = game['prob_home']
        prob_away = 1 - prob_home
        
        vegas_ml_home = game['home_moneyline']
        vegas_ml_away = game['away_moneyline']
        
        if vegas_ml_home < 0: v_prob_home = -vegas_ml_home/(-vegas_ml_home+100)
        else: v_prob_home = 100/(vegas_ml_home+100)
        
        if vegas_ml_away < 0: v_prob_away = -vegas_ml_away/(-vegas_ml_away+100)
        else: v_prob_away = 100/(vegas_ml_away+100)
        
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
            
        if is_push: profit = 0
        elif did_win:
             if pick_odds > 0: decimal = (pick_odds/100) + 1
             else: decimal = (100/abs(pick_odds)) + 1
             profit = pick_units * (decimal - 1)
        else:
            profit = -pick_units
            
        total_profit += profit
        total_bets += 1
        
    trial.set_user_attr("bets", total_bets)
    trial.set_user_attr("profit", total_profit)
    
    # CONSTRAINT: Must have at least MIN_BETS
    if total_bets < MIN_BETS:
        return -100.0  # Heavy penalty
    
    # OBJECTIVE: Maximize ROI (profit per bet)
    roi = total_profit / total_bets
    return roi

# --- PRE-CALCULATION ---
print("Loading Data for Sniper Mode Optimization...")
with open(CACHE_PATH_V2, 'rb') as f: db = pickle.load(f)
with open(MODEL_ML_PATH, 'rb') as f: ml_model = pickle.load(f)
with open(MODEL_ML_FEATURES, 'rb') as f: ml_features = pickle.load(f)

schedule = db['schedule']
completed = schedule[(schedule['result'].notna()) & (schedule['game_type'] == 'REG') & (schedule['season'] == 2025)].copy()

print("Pre-calculating predictions...")
full_games = v2_features.engineering_pipeline(db)
games_with_features = full_games[full_games['game_id'].isin(completed['game_id'])].copy()
games_with_features = pd.merge(games_with_features, completed[['game_id', 'home_score', 'away_score']], on='game_id', how='left', suffixes=('', '_y'))
games_with_features = games_with_features.dropna(subset=['home_moneyline', 'away_moneyline'])

for c in ml_features:
    if c not in games_with_features.columns: games_with_features[c] = 0

X = games_with_features[ml_features]
games_with_features['prob_home'] = ml_model.predict_proba(X)[:, 1]

SIM_GAMES = games_with_features
print(f"Optimization Set: {len(SIM_GAMES)} games (2025 season only)")

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')  # Maximize ROI
    study.optimize(objective, n_trials=200)

    print("\n--- 🎯 SNIPER MODE RESULTS 🎯 ---")
    print(f"Best ROI: {study.best_value:.4f}u per bet")
    
    best = study.best_trial
    print(f"Bets: {best.user_attrs.get('bets', 0)}")
    print(f"Profit: {best.user_attrs.get('profit', 0):.2f}u")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            
    # Save best params
    import json
    with open("config/best_ml_sniper.json", "w") as f:
        result = study.best_params.copy()
        result['bets'] = best.user_attrs.get('bets', 0)
        result['profit'] = best.user_attrs.get('profit', 0)
        result['roi'] = study.best_value
        json.dump(result, f, indent=2)
    print("\nSaved to best_ml_sniper.json")
