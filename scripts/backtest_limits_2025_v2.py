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
    units = min(max(0, kelly_fraction_val * KELLY_FRACTION), MAX_UNITS) # 0.5 Kelly
    return units

def evaluate_params(games_df, params):
    total_profit = 0.0
    total_bets = 0
    wins = 0
    losses = 0
    pushes = 0
    
    for _, game in games_df.iterrows():
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
        
        pick_units = 0; pick_odds = 0; did_win = False; is_push = False
        
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
            
        if is_push: 
            profit = 0
            pushes += 1
        elif did_win:
             if pick_odds > 0: decimal = (pick_odds/100) + 1
             else: decimal = (100/abs(pick_odds)) + 1
             profit = pick_units * (decimal - 1)
             wins += 1
        else:
            profit = -pick_units
            losses += 1
            
        total_profit += profit
        total_bets += 1
        
    roi = total_profit / total_bets if total_bets > 0 else 0
    return total_profit, total_bets, roi, f"{wins}-{losses}-{pushes}"


# --- LOAD ---
with open(CACHE_PATH_V2, 'rb') as f: db = pickle.load(f)
ml_model = xgb.XGBClassifier()
ml_model.load_model(MODEL_ML_PATH)

schedule = db['schedule']
completed = schedule[(schedule['result'].notna()) & (schedule['game_type'] == 'REG') & (schedule['season'] == 2025)].copy()

if completed.empty:
    print("No completed 2025 games found!")
    exit()

# Features
full_games = v2_features.engineering_pipeline(db)
games = full_games[full_games['game_id'].isin(completed['game_id'])].copy()
games = pd.merge(games, completed[['game_id', 'home_score', 'away_score']], on='game_id', how='left', suffixes=('', '_y'))
games = games.dropna(subset=['home_moneyline', 'away_moneyline'])

try: ml_cols = ml_model.feature_names
except: ml_cols = ml_model.get_booster().feature_names

for c in ml_cols:
    if c not in games.columns: games[c] = 0
    
X = games[ml_cols]
games['prob_home'] = ml_model.predict_proba(X)[:, 1]

# --- TEST CASES ---
test_cases = [
    # Symmetric
    {"label": "+/- 220", "min": -220, "max": 220},
    {"label": "+/- 301", "min": -301, "max": 301},
    {"label": "+/- 438", "min": -438, "max": 438},
    {"label": "+/- 472", "min": -472, "max": 472},
    {"label": "+/- 500", "min": -500, "max": 500},
    # Asymmetric (From Optimization)
    {"label": "Asym (High Val)", "min": -407, "max": 209},
    {"label": "Asym (Balanced)", "min": -441, "max": 302},
    {"label": "Asym (Max Profit)", "min": -472, "max": 331},
]

base_params = {
    "min_edge": 0.0486, 
    "kelly_fraction": 0.50 
}

print(f"{'Limit Cap':<18} | {'Bets':<6} | {'Profit':<8} | {'ROI':<8}")
print("-" * 50)

results = []
for case in test_cases:
    params = base_params.copy()
    params['min_odds_limit'] = case['min']
    params['max_odds_limit'] = case['max']
    
    profit, bets, roi, record = evaluate_params(games, params)
    
    # Store for CSV
    results.append({
        'Limit Cap': case['label'],
        'Bets': bets,
        'Profit': round(profit, 2),
        'ROI': round(roi, 3)
    })
    
    print(f"{case['label']:<18} | {bets:<6} | {profit:>6.2f}u | {roi:>6.3f}u")

pd.DataFrame(results).to_csv("backtest_2025_asym.csv", index=False)
