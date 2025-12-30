import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from scipy.stats import norm
import src.features as v2_features

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_SPREAD_PATH = "models/v2_spread.json"
MODEL_TOTAL_PATH = "models/v2_total.json"
MODEL_ML_PATH = "models/v2_moneyline.json"
HISTORY_PATH = "data/betting_history.csv"
BACKUP_PATH = "data/betting_history_backup.csv"

# --- KELLY LOGIC (V3.2 - Balanced Volume/Profit) ---
def calculate_kelly_units(abs_edge):
    """Calculates Kelly Unit size based on point spread edge."""
    STD_DEV = 12.53 
    PAYOUT_RATIO = 0.9091
    # V3.2: Balanced from sweep - 3.5 optimal for volume with profit
    MIN_EDGE = 3.5
    MAX_UNITS = 5.0
    KELLY_FRACTION = 0.10  # V3.2: More conservative

    if abs_edge < MIN_EDGE:
        return 0.0, "None"

    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    units = round(min(kelly_units, MAX_UNITS), 1)
    
    if units >= 1.5: conf = "STRONG"
    elif units >= 0.8: conf = "SOLID" 
    elif units >= 0.1: conf = "LEAN"
    else: conf = "None"
        
    return units, conf

def calculate_totals_kelly(abs_edge):
    """Calculates Kelly Unit size for TOTALS."""
    STD_DEV = 13.5
    PAYOUT_RATIO = 0.9091
    # V3.2: Confirmed optimal
    MIN_EDGE = 4.8
    MAX_UNITS = 4.8
    KELLY_FRACTION = 0.21

    if abs_edge < MIN_EDGE:
        return 0.0, "None"

    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    units = round(min(kelly_units, MAX_UNITS), 1)
    
    if units >= 0.8: conf = "STRONG"
    elif units >= 0.5: conf = "SOLID" 
    elif units >= 0.1: conf = "LEAN"
    else: conf = "None"
        
    return units, conf

def calculate_moneyline_kelly(model_prob, vegas_prob, vegas_odds):
    """Calculates Kelly Unit size for MONEYLINE using probability-based EV."""
    # V3.2: Balanced 5% edge threshold
    MIN_EDGE = 0.05
    MAX_UNITS = 3.0
    
    edge = model_prob - vegas_prob
    
    if edge < MIN_EDGE:
        return 0.0, "None"
    
    if vegas_odds < 0:
        payout = 100 / abs(vegas_odds)
    else:
        payout = vegas_odds / 100
    
    b = payout
    p = model_prob
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b if b > 0 else 0
    # V3.2: Kelly fraction 0.12
    units = round(min(max(0, kelly_fraction * 0.12), MAX_UNITS), 1)
    
    if units > 0:
        conf = "SOLID" if edge >= 0.10 else "LEAN"
    else:
        conf = "None"
        
    return units, conf

def run_simulation():
    print("🚀 Starting Protocol 705 V3.2 Simulation...")
    
    # 1. Load Data
    if not os.path.exists(CACHE_PATH_V2):
        print("❌ Cache not found!")
        return

    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    
    schedule = db['schedule']
    base_stats = db['base_stats']
    injury_stats = db['injury_stats']
    
    # 2. Load Models
    print("Loading V3.2 Models...")
    spread_model = xgb.Booster()
    spread_model.load_model(MODEL_SPREAD_PATH)
    total_model = xgb.Booster()
    total_model.load_model(MODEL_TOTAL_PATH)
    ml_model = xgb.XGBClassifier()
    ml_model.load_model(MODEL_ML_PATH)
    
    # Get Feature Names
    spread_features = spread_model.feature_names
    total_features = total_model.feature_names
    try:
        ml_features = ml_model.feature_names
    except:
        ml_features = ml_model.get_booster().feature_names

    # 3. Determine Weeks to Simulate
    # We want to simulate all completed games in the current season (where result is not NaN)
    completed_games = schedule[(schedule['result'].notna()) & (schedule['game_type'] == 'REG')].copy()
    if completed_games.empty:
        print("No completed games to simulate.")
        return

    current_season = completed_games['season'].max()
    weeks = sorted(completed_games[completed_games['season'] == current_season]['week'].unique())
    
    print(f"Simulating Season {current_season} | Weeks: {weeks}")
    
    history_records = []
    
    # Run Full Engineering Pipeline ONCE to save time, then filter by week
    # Note: treating "future" info is tricky. The pipeline uses "shift(1)" so it SHOULD be safe 
    # if we pass the full schedule.
    print("Running Engineering Pipeline...")
    full_games = v2_features.engineering_pipeline(schedule, base_stats, injury_stats)
    
    total_profit = 0.0
    
    for week in weeks:
        print(f"   Simulating Week {week}...")
        
        # Get games for this week
        week_games = full_games[(full_games['season'] == current_season) & (full_games['week'] == week)].copy()
        
        for _, game in week_games.iterrows():
            # Only grade completed games
            # We need to find the specific game result in 'schedule' to get the actual score
            # because engineering_pipeline might rely on base_stats which might not have the score yet?
            # Actually engineering_pipeline merges betting_df which has 'su_win' but maybe not scores directly in easy format.
            # Let's look up the result in the raw schedule df.
            real_res = schedule[(schedule['game_id'] == game['game_id'])].iloc[0]
            
            # Check if home_score/away_score present (should be if result is not nan)
            if pd.isna(real_res['home_score']) or pd.isna(real_res['away_score']):
                continue
                
            home_score = real_res['home_score']
            away_score = real_res['away_score']
            
            # --- PREDICTION ---
            row_df = pd.DataFrame([game])
            
            # Spread Pred
            for c in spread_features:
                if c not in row_df.columns: row_df[c] = 0
            X_spread = xgb.DMatrix(row_df[spread_features])
            pred_margin = spread_model.predict(X_spread)[0] # Home Margin
            
            # Total Pred
            for c in total_features:
                if c not in row_df.columns: row_df[c] = 0
            X_total = xgb.DMatrix(row_df[total_features])
            pred_total = total_model.predict(X_total)[0]
            
            # ML Pred
            for c in ml_features:
                if c not in row_df.columns: row_df[c] = 0
            X_ml = row_df[ml_features]
            prob_home_win = ml_model.predict_proba(X_ml)[0][1]
            
            # --- BETTING LOGIC & GRADING ---
            
            # 1. SPREAD
            raw_away_spread = real_res['spread_line']
            if not pd.isna(raw_away_spread):
                # Calculate Expectation
                home_spread = -1 * raw_away_spread
                vegas_home_margin = -1 * home_spread
                edge_spread = pred_margin - vegas_home_margin
                
                units, conf = calculate_kelly_units(abs(edge_spread))
                
                if units > 0:
                    pick_team = game['home_team'] if edge_spread > 0 else game['away_team']
                    
                    # Grading
                    actual_margin = home_score - away_score
                    # If pick Home (edge > 0): Win if actual_margin > vegas_home_margin (Cover)
                    # Example: Vegas Home Margin +7 (Spread -7). Actual +10. Win.
                    
                    # Logic: 
                    # Spread result is usually: Did Home Cover? 
                    # Home Cover if (Home Score + Home Spread) > Away Score
                    # => Home Margin > -Home Spread
                    # => Home Margin > Vegas Expected Home Margin?
                    # Wait. Home Spread -7. Home Score 24, Away 10. Margin 14.
                    # 24 + (-7) = 17 > 10. Cover.
                    # 14 > 7. Correct.
                    
                    # Determine Result
                    profit = 0.0
                    result = "PUSH"
                    
                    covered = False
                    push = False
                    
                    if pick_team == game['home_team']:
                        if actual_margin > vegas_home_margin: covered = True
                        elif actual_margin == vegas_home_margin: push = True
                    else:
                        # Pick Away
                        if actual_margin < vegas_home_margin: covered = True
                        elif actual_margin == vegas_home_margin: push = True
                        
                    if push:
                        result = "PUSH"
                        profit = 0.0
                    elif covered:
                        result = "WIN"
                        profit = units * 0.9091 # Standard -110
                    else:
                        result = "LOSS"
                        profit = -units
                    
                    total_profit += profit
                    
                    history_records.append({
                        'season': current_season,
                        'week': week,
                        'game_id': game['game_id'],
                        'matchup': f"{game['away_team']} @ {game['home_team']}",
                        'home': game['home_team'],
                        'away': game['away_team'],
                        'type': 'spread',
                        'pick_team': pick_team,
                        'units': units,
                        'status': 'GRADED',
                        'result': result,
                        'profit': round(profit, 2)
                    })

            # 2. TOTAL
            raw_total = real_res['total_line']
            if not pd.isna(raw_total):
                fair_total = max(min(pred_total, 70), 20)
                diff_total = fair_total - raw_total
                abs_total_edge = abs(diff_total)
                units_total, _ = calculate_totals_kelly(abs_total_edge)  # USE NEW TOTALS KELLY
                
                if units_total > 0:
                    pick_type = "OVER" if diff_total > 0 else "UNDER"
                    actual_total = home_score + away_score
                    
                    profit = 0.0
                    result = "PUSH"
                    
                    win = False
                    push = False
                    
                    if actual_total == raw_total:
                        push = True
                    elif pick_type == "OVER":
                        if actual_total > raw_total: win = True
                    else: # UNDER
                        if actual_total < raw_total: win = True
                        
                    if push:
                        result = "PUSH"
                        profit = 0.0
                    elif win:
                        result = "WIN"
                        profit = units_total * 0.9091
                    else:
                        result = "LOSS"
                        profit = -units_total
                        
                    total_profit += profit
                    
                    history_records.append({
                        'season': current_season,
                        'week': week,
                        'game_id': game['game_id'],
                        'matchup': f"{game['away_team']} @ {game['home_team']}",
                        'home': game['home_team'],
                        'away': game['away_team'],
                        'type': 'total',
                        'pick_team': pick_type,
                        'units': units_total,
                        'status': 'GRADED',
                        'result': result,
                        'profit': round(profit, 2)
                    })

            # 3. MONEYLINE (V3.1 - Evaluate BOTH Home and Away)
            if not pd.isna(real_res.get('home_moneyline')) and not pd.isna(real_res.get('away_moneyline')):
                vegas_ml_home = real_res['home_moneyline']
                vegas_ml_away = real_res['away_moneyline']
                
                prob_away_win = 1 - prob_home_win
                
                # Vegas Implied Probabilities
                if vegas_ml_home < 0: v_prob_home = -vegas_ml_home / (-vegas_ml_home + 100)
                else: v_prob_home = 100 / (vegas_ml_home + 100)
                
                if vegas_ml_away < 0: v_prob_away = -vegas_ml_away / (-vegas_ml_away + 100)
                else: v_prob_away = 100 / (vegas_ml_away + 100)
                
                # Calculate Kelly for BOTH sides
                units_home, _ = calculate_moneyline_kelly(prob_home_win, v_prob_home, vegas_ml_home)
                units_away, _ = calculate_moneyline_kelly(prob_away_win, v_prob_away, vegas_ml_away)
                
                # Pick the better opportunity
                if units_home > 0 and units_home >= units_away:
                    units_ml = units_home
                    pick_team = game['home_team']
                    pick_odds = vegas_ml_home
                    did_win = home_score > away_score
                elif units_away > 0:
                    units_ml = units_away
                    pick_team = game['away_team']
                    pick_odds = vegas_ml_away
                    did_win = away_score > home_score
                else:
                    units_ml = 0  # No bet
                    pick_team = None
                
                if units_ml > 0 and pick_team:
                    # Result
                    profit = 0.0
                    
                    if home_score == away_score:
                        result = "PUSH"
                        profit = 0.0
                    elif did_win:
                        result = "WIN"
                        if pick_odds > 0:
                            decimal_odds = (pick_odds / 100) + 1
                        else:
                            decimal_odds = (100 / abs(pick_odds)) + 1
                        profit = units_ml * (decimal_odds - 1)
                    else:
                        result = "LOSS"
                        profit = -units_ml
                        
                    total_profit += profit
                    
                    history_records.append({
                        'season': current_season,
                        'week': week,
                        'game_id': game['game_id'],
                        'matchup': f"{game['away_team']} @ {game['home_team']}",
                        'home': game['home_team'],
                        'away': game['away_team'],
                        'type': 'moneyline',
                        'pick_team': pick_team + " ML",
                        'units': units_ml,
                        'status': 'GRADED',
                        'result': result,
                        'profit': round(profit, 2)
                    })

    # 4. Save
    print(f"Simulation Complete. Total Simulated Profit: {total_profit:.2f}u")
    
    # Backup
    if os.path.exists(HISTORY_PATH):
        try:
            if os.path.exists(BACKUP_PATH):
                os.remove(BACKUP_PATH) # Remove old backup
            os.rename(HISTORY_PATH, BACKUP_PATH)
            print(f"Backed up old history to {BACKUP_PATH}")
        except OSError as e:
            print(f"⚠️ Could not backup history (file locked?): {e}")
            print("   Overwriting existing history file...")
        
    # Save New
    df = pd.DataFrame(history_records)
    if not df.empty:
        df.to_csv(HISTORY_PATH, index=False)
        print(f"✅ Saved {len(df)} records to {HISTORY_PATH}")
    else:
        print("⚠️ No bets made in simulation.")

if __name__ == "__main__":
    run_simulation()
