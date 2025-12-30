"""
Optimize unit sizing parameters for all bet types (Spread, Total, Moneyline)
using Optuna to maximize ROI while maintaining reasonable unit limits.
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

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_SPREAD_PATH = "models/v4_ensemble_stack.pkl"
MODEL_SPREAD_FEATURES = "models/v4_features.pkl"
MODEL_TOTAL_PATH = "models/v4_total_stack.pkl"
MODEL_TOTAL_FEATURES = "models/v4_total_features.pkl"
MODEL_ML_PATH = "models/v4_moneyline_stack.pkl"
MODEL_ML_FEATURES = "models/v4_moneyline_features.pkl"

# Reasonable unit limits (bankroll management)
MAX_SINGLE_BET = 3.0  # Never bet more than 3u on a single play
MIN_BETS_SPREAD = 15  # Minimum bets per season
MIN_BETS_TOTAL = 10
MIN_BETS_ML = 10

# Pre-calculated data (global for optimization)
SIM_GAMES = None
SPREAD_PREDS = None
TOTAL_PREDS = None
ML_PREDS = None

def calculate_spread_kelly(abs_edge, params):
    """Spread Kelly with parameterized values."""
    STD_DEV = 12.53
    PAYOUT_RATIO = 0.9091
    
    if abs_edge < params['min_edge']:
        return 0.0
    
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score)
    q = 1.0 - p
    
    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * params['kelly_fraction'])
    return min(kelly_units, params['max_units'])

def calculate_total_kelly(abs_edge, params):
    """Totals Kelly with parameterized values."""
    STD_DEV = 13.5
    PAYOUT_RATIO = 0.9091
    
    if abs_edge < params['min_edge']:
        return 0.0
    
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score)
    q = 1.0 - p
    
    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * params['kelly_fraction'])
    return min(kelly_units, params['max_units'])

def calculate_ml_kelly(prob, vegas_prob, vegas_odds, params):
    """Moneyline Kelly with parameterized values."""
    # Odds Filter
    if vegas_odds < params['min_odds'] or vegas_odds > params['max_odds']:
        return 0.0
    
    edge = prob - vegas_prob
    if edge < params['min_edge']:
        return 0.0
    
    if vegas_odds < 0:
        payout = 100 / abs(vegas_odds)
    else:
        payout = vegas_odds / 100
    
    b = payout
    p = prob
    q = 1 - p
    
    kelly_fraction_val = (b * p - q) / b if b > 0 else 0
    units = min(max(0, kelly_fraction_val * params['kelly_fraction']), params['max_units'])
    return units

def objective_spread(trial):
    params = {
        'min_edge': trial.suggest_float('min_edge', 2.0, 6.0),
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.05, 0.25),
        'max_units': trial.suggest_float('max_units', 1.0, MAX_SINGLE_BET)
    }
    
    total_profit = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        pred_margin = game['spread_pred']
        raw_away_spread = game['spread_line']
        
        if pd.isna(raw_away_spread):
            continue
            
        home_spread = -1 * raw_away_spread
        vegas_home_margin = -1 * home_spread
        edge = pred_margin - vegas_home_margin
        
        units = calculate_spread_kelly(abs(edge), params)
        
        if units > 0:
            pick_team = game['home_team'] if edge > 0 else game['away_team']
            actual_margin = game['home_score'] - game['away_score']
            
            # Grading
            if pick_team == game['home_team']:
                covered = actual_margin > vegas_home_margin
                push = actual_margin == vegas_home_margin
            else:
                covered = actual_margin < vegas_home_margin
                push = actual_margin == vegas_home_margin
            
            if push:
                profit = 0.0
            elif covered:
                profit = units * 0.9091
            else:
                profit = -units
            
            total_profit += profit
            total_bets += 1
    
    trial.set_user_attr("bets", total_bets)
    trial.set_user_attr("profit", total_profit)
    
    if total_bets < MIN_BETS_SPREAD:
        return -100.0
    
    return total_profit / total_bets  # ROI

def objective_total(trial):
    params = {
        'min_edge': trial.suggest_float('min_edge', 3.0, 8.0),
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.05, 0.30),
        'max_units': trial.suggest_float('max_units', 1.0, MAX_SINGLE_BET)
    }
    
    total_profit = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        pred_total = game['total_pred']
        raw_total = game['total_line']
        
        if pd.isna(raw_total):
            continue
        
        fair_total = max(min(pred_total, 70), 20)
        diff_total = fair_total - raw_total
        abs_edge = abs(diff_total)
        
        units = calculate_total_kelly(abs_edge, params)
        
        if units > 0:
            pick_type = "OVER" if diff_total > 0 else "UNDER"
            actual_total = game['home_score'] + game['away_score']
            
            push = actual_total == raw_total
            if pick_type == "OVER":
                win = actual_total > raw_total
            else:
                win = actual_total < raw_total
            
            if push:
                profit = 0.0
            elif win:
                profit = units * 0.9091
            else:
                profit = -units
            
            total_profit += profit
            total_bets += 1
    
    trial.set_user_attr("bets", total_bets)
    trial.set_user_attr("profit", total_profit)
    
    if total_bets < MIN_BETS_TOTAL:
        return -100.0
    
    return total_profit / total_bets  # ROI

def objective_ml(trial):
    params = {
        'min_edge': trial.suggest_float('min_edge', 0.08, 0.25),
        'kelly_fraction': trial.suggest_float('kelly_fraction', 0.2, 1.5),
        'max_units': trial.suggest_float('max_units', 0.5, MAX_SINGLE_BET),
        'min_odds': trial.suggest_int('min_odds', -500, -150),
        'max_odds': trial.suggest_int('max_odds', 100, 400)
    }
    
    total_profit = 0.0
    total_bets = 0
    
    for _, game in SIM_GAMES.iterrows():
        prob_home = game['ml_pred']
        prob_away = 1 - prob_home
        
        vegas_ml_home = game.get('home_moneyline')
        vegas_ml_away = game.get('away_moneyline')
        
        if pd.isna(vegas_ml_home) or pd.isna(vegas_ml_away):
            continue
        
        # Vegas implied probabilities
        if vegas_ml_home < 0:
            v_prob_home = -vegas_ml_home / (-vegas_ml_home + 100)
        else:
            v_prob_home = 100 / (vegas_ml_home + 100)
        
        if vegas_ml_away < 0:
            v_prob_away = -vegas_ml_away / (-vegas_ml_away + 100)
        else:
            v_prob_away = 100 / (vegas_ml_away + 100)
        
        units_home = calculate_ml_kelly(prob_home, v_prob_home, vegas_ml_home, params)
        units_away = calculate_ml_kelly(prob_away, v_prob_away, vegas_ml_away, params)
        
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
        
        is_push = game['home_score'] == game['away_score']
        
        if is_push:
            profit = 0.0
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
    
    trial.set_user_attr("bets", total_bets)
    trial.set_user_attr("profit", total_profit)
    
    if total_bets < MIN_BETS_ML:
        return -100.0
    
    return total_profit / total_bets  # ROI

def load_data():
    global SIM_GAMES
    
    print("Loading data...")
    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    
    # Load all models
    with open(MODEL_SPREAD_PATH, 'rb') as f:
        spread_model = pickle.load(f)
    with open(MODEL_SPREAD_FEATURES, 'rb') as f:
        spread_features = pickle.load(f)
    
    with open(MODEL_TOTAL_PATH, 'rb') as f:
        total_model = pickle.load(f)
    with open(MODEL_TOTAL_FEATURES, 'rb') as f:
        total_features = pickle.load(f)
    
    with open(MODEL_ML_PATH, 'rb') as f:
        ml_model = pickle.load(f)
    with open(MODEL_ML_FEATURES, 'rb') as f:
        ml_features = pickle.load(f)
    
    # Get completed games
    schedule = db['schedule']
    completed = schedule[
        (schedule['result'].notna()) & 
        (schedule['game_type'] == 'REG') & 
        (schedule['season'] == 2025)
    ].copy()
    
    print("Running feature engineering...")
    full_games = v2_features.engineering_pipeline(db)
    games = full_games[full_games['game_id'].isin(completed['game_id'])].copy()
    
    # Merge scores
    games = pd.merge(
        games,
        completed[['game_id', 'home_score', 'away_score']],
        on='game_id', how='left', suffixes=('', '_y')
    )
    
    # Fill missing features
    for c in spread_features:
        if c not in games.columns:
            games[c] = 0
    for c in total_features:
        if c not in games.columns:
            games[c] = 0
    for c in ml_features:
        if c not in games.columns:
            games[c] = 0
    
    # Make predictions
    print("Generating predictions...")
    games['spread_pred'] = spread_model.predict(games[spread_features])
    games['total_pred'] = total_model.predict(games[total_features])
    games['ml_pred'] = ml_model.predict_proba(games[ml_features])[:, 1]
    
    SIM_GAMES = games
    print(f"Loaded {len(SIM_GAMES)} games for optimization")

if __name__ == "__main__":
    load_data()
    
    print("\n" + "="*60)
    print("🎯 SPREAD OPTIMIZATION (Max Units: 3.0u)")
    print("="*60)
    study_spread = optuna.create_study(direction='maximize')
    study_spread.optimize(objective_spread, n_trials=150)
    
    print(f"\nBest Spread ROI: {study_spread.best_value:.4f}u/bet")
    best_s = study_spread.best_trial
    print(f"Bets: {best_s.user_attrs.get('bets')}, Profit: {best_s.user_attrs.get('profit'):.2f}u")
    print("Parameters:")
    for k, v in study_spread.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("📊 TOTALS OPTIMIZATION (Max Units: 3.0u)")
    print("="*60)
    study_total = optuna.create_study(direction='maximize')
    study_total.optimize(objective_total, n_trials=150)
    
    print(f"\nBest Totals ROI: {study_total.best_value:.4f}u/bet")
    best_t = study_total.best_trial
    print(f"Bets: {best_t.user_attrs.get('bets')}, Profit: {best_t.user_attrs.get('profit'):.2f}u")
    print("Parameters:")
    for k, v in study_total.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("💰 MONEYLINE OPTIMIZATION (Max Units: 3.0u)")
    print("="*60)
    study_ml = optuna.create_study(direction='maximize')
    study_ml.optimize(objective_ml, n_trials=150)
    
    print(f"\nBest ML ROI: {study_ml.best_value:.4f}u/bet")
    best_m = study_ml.best_trial
    print(f"Bets: {best_m.user_attrs.get('bets')}, Profit: {best_m.user_attrs.get('profit'):.2f}u")
    print("Parameters:")
    for k, v in study_ml.best_params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    import json
    results = {
        'spread': {
            **study_spread.best_params,
            'bets': best_s.user_attrs.get('bets'),
            'profit': best_s.user_attrs.get('profit'),
            'roi': study_spread.best_value
        },
        'total': {
            **study_total.best_params,
            'bets': best_t.user_attrs.get('bets'),
            'profit': best_t.user_attrs.get('profit'),
            'roi': study_total.best_value
        },
        'moneyline': {
            **study_ml.best_params,
            'bets': best_m.user_attrs.get('bets'),
            'profit': best_m.user_attrs.get('profit'),
            'roi': study_ml.best_value
        }
    }
    
    with open("config/optimal_unit_params.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved all results to optimal_unit_params.json")
