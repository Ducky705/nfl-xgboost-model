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
    MIN_ODDS = params['min_odds_limit'] # Negative value e.g. -400
    MAX_ODDS = params['max_odds_limit'] # Positive value e.g. +250
    MAX_UNITS = 3.0
    
    # 1. Odds Filter (Asymmetric)
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
    # We want to sweep the limit threshold specifically
    params = {
        'min_edge': 0.0486, # Fixed from previous opt
        'kelly_fraction': 0.50, # Fixed from previous opt
        'min_odds_limit': trial.suggest_int('min_odds_limit', -1000, -110),
        'max_odds_limit': trial.suggest_int('max_odds_limit', 100, 1000)
    }
    
    total_profit = 0.0
    total_bets = 0
    
    # Iterate through games
    for _, game in SIM_GAMES.iterrows():
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
        
    # --- MULTI-OBJECTIVE ---
    if total_bets < 20: 
        return -100.0, -1.0
        
    roi = total_profit / total_bets
    
    trial.set_user_attr("bets", total_bets)
    
    return total_profit, roi

# --- PRE-CALCULATION ---
print("Loading Data for Asymmetric Optimization...")
db = load_data()
ml_model = load_model()

schedule = db['schedule']
completed = schedule[(schedule['result'].notna()) & (schedule['game_type'] == 'REG')].copy()

full_games = v2_features.engineering_pipeline(db)
games_with_features = full_games[full_games['game_id'].isin(completed['game_id'])].copy()
games_with_features = pd.merge(games_with_features, completed[['game_id', 'home_score', 'away_score']], on='game_id', how='left', suffixes=('', '_y'))
games_with_features = games_with_features.dropna(subset=['home_moneyline', 'away_moneyline'])

try: ml_cols = ml_model.feature_names
except: ml_cols = ml_model.get_booster().feature_names

for c in ml_cols:
    if c not in games_with_features.columns: games_with_features[c] = 0

X = games_with_features[ml_cols]
games_with_features['prob_home'] = ml_model.predict_proba(X)[:, 1]

SIM_GAMES = games_with_features
print(f"Optimization Set: {len(SIM_GAMES)} games")

if __name__ == "__main__":
    # Maximize Profit AND ROI
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=200)

    print("\n--- 🏆 ASYMMETRIC PARETO FRONTIER 🏆 ---")
    print(f"{'Trial':<6} | {'Profit':<8} | {'ROI':<8} | {'Bets':<5} | {'Min (Fav)':<10} | {'Max (Dog)':<10}")
    print("-" * 75)
    
    best_trials = study.best_trials
    best_trials.sort(key=lambda x: x.values[0], reverse=True)
    
    for trial in best_trials:
        p1 = trial.values[0] # Profit
        p2 = trial.values[1] # ROI
        bets = trial.user_attrs.get("bets", 0)
        params = trial.params
        
        lim_min = params['min_odds_limit']
        lim_max = params['max_odds_limit']
        
        print(f"{trial.number:<6} | {p1:<8.2f} | {p2:<8.3f} | {bets:<5} | {lim_min:<10} | {lim_max:<10}")

    results = []
    for trial in best_trials:
        d = trial.params.copy()
        d['profit'] = trial.values[0]
        d['roi'] = trial.values[1]
        d['bets'] = trial.user_attrs.get("bets", 0)
        results.append(d)
        
    pd.DataFrame(results).to_csv("asymmetric_pareto_results.csv", index=False)
