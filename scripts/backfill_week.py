"""
Backfill Week 18 Picks - Retroactive Generation and Grading
"""
import pickle
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

import src.features as v2_features
from src.ensemble_model import VotingEnsemble
from scipy.stats import norm
import numpy as np

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
DATA_PATH = "data/betting_history.csv"
MODEL_STACK_PATH = "models/v3_ensemble_stack.pkl"
MODEL_TOTAL_PATH = "models/v3_total_stack.pkl"
MODEL_ML_PATH = "models/v3_moneyline_stack.pkl"

# Load Kelly functions from main.py logic
def calculate_kelly_units(abs_edge):
    STD_DEV = 12.53
    PAYOUT_RATIO = 0.9091
    MIN_EDGE = 3.5
    MAX_UNITS = 2.0
    KELLY_FRACTION = 0.035
    
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
    STD_DEV = 13.5
    PAYOUT_RATIO = 0.9091
    MIN_EDGE = 4.5
    MAX_UNITS = 2.0
    KELLY_FRACTION = 0.05
    
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
    MIN_EDGE = 0.0503
    MAX_UNITS = 2.0
    MIN_ODDS = -134
    MAX_ODDS = 127
    KELLY_FRACTION = 2.91
    
    if vegas_odds < MIN_ODDS or vegas_odds > MAX_ODDS:
        return 0.0, "None"
    
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
    units = round(min(max(0, kelly_fraction * KELLY_FRACTION), MAX_UNITS), 1)
    
    if units > 0:
        conf = "SOLID" if edge >= 0.15 else "LEAN"
    else:
        conf = "None"
    
    return units, conf

def grade_pick(pick, schedule_row):
    """Grade a pick against actual results."""
    home_score = schedule_row['home_score']
    away_score = schedule_row['away_score']
    
    if pd.isna(home_score) or pd.isna(away_score):
        return "PENDING", 0.0
    
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    
    bet_type = pick['type']
    units = pick['units']
    
    if bet_type == 'spread':
        pick_team = pick['pick']
        spread_line = schedule_row['spread_line']  # Away team spread
        home_spread = -1 * spread_line
        
        if pick_team == schedule_row['home_team']:
            # Bet on home team
            if actual_margin > home_spread:
                return "WIN", units * 0.91
            elif actual_margin < home_spread:
                return "LOSS", -units
            else:
                return "PUSH", 0.0
        else:
            # Bet on away team
            if actual_margin < home_spread:
                return "WIN", units * 0.91
            elif actual_margin > home_spread:
                return "LOSS", -units
            else:
                return "PUSH", 0.0
                
    elif bet_type == 'total':
        pick_dir = pick['pick']  # "OVER" or "UNDER"
        total_line = schedule_row['total_line']
        
        if pick_dir == "OVER":
            if actual_total > total_line:
                return "WIN", units * 0.91
            elif actual_total < total_line:
                return "LOSS", -units
            else:
                return "PUSH", 0.0
        else:  # UNDER
            if actual_total < total_line:
                return "WIN", units * 0.91
            elif actual_total > total_line:
                return "LOSS", -units
            else:
                return "PUSH", 0.0
                
    elif bet_type == 'moneyline':
        pick_team = pick['pick'].replace(' ML', '')
        
        if pick_team == schedule_row['home_team']:
            if home_score > away_score:
                # Calculate profit based on odds
                odds = schedule_row['home_moneyline']
                if odds < 0:
                    profit = units * (100 / abs(odds))
                else:
                    profit = units * (odds / 100)
                return "WIN", round(profit, 2)
            else:
                return "LOSS", -units
        else:  # Away team
            if away_score > home_score:
                odds = schedule_row['away_moneyline']
                if odds < 0:
                    profit = units * (100 / abs(odds))
                else:
                    profit = units * (odds / 100)
                return "WIN", round(profit, 2)
            else:
                return "LOSS", -units
    
    return "PENDING", 0.0


if __name__ == "__main__":
    WEEK_TO_BACKFILL = 18
    
    print(f"=== Backfilling Week {WEEK_TO_BACKFILL} Picks ===")
    
    # Load DB and models
    print("Loading database...")
    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    
    print("Loading models...")
    with open(MODEL_STACK_PATH, 'rb') as f:
        spread_model = pickle.load(f)
    with open("models/v3_features.pkl", 'rb') as f:
        v4_features = pickle.load(f)
    with open(MODEL_TOTAL_PATH, 'rb') as f:
        total_model = pickle.load(f)
    with open("models/v3_total_features.pkl", 'rb') as f:
        v4_total_features = pickle.load(f)
    with open(MODEL_ML_PATH, 'rb') as f:
        ml_model = pickle.load(f)
    with open("models/v3_moneyline_features.pkl", 'rb') as f:
        v4_ml_features = pickle.load(f)
    
    # Get current season
    schedule = db['schedule']
    CURRENT_SEASON = schedule['season'].max()
    
    # Run feature engineering
    print("Running feature engineering...")
    full_games = v2_features.engineering_pipeline(db)
    
    # Filter for Week 18
    week_df = full_games[(full_games['season'] == CURRENT_SEASON) & (full_games['week'] == WEEK_TO_BACKFILL)].copy()
    print(f"Found {len(week_df)} games in Week {WEEK_TO_BACKFILL}")
    
    if week_df.empty:
        print("No games found for this week!")
        exit(1)
    
    # Generate predictions
    new_picks = []
    
    for _, game in week_df.iterrows():
        row_df = pd.DataFrame([game])
        
        # Spread prediction
        for c in v4_features:
            if c not in row_df.columns:
                row_df[c] = 0
        X_spread = row_df[v4_features]
        pred_margin = spread_model.predict(X_spread)[0]
        
        # Total prediction
        for c in v4_total_features:
            if c not in row_df.columns:
                row_df[c] = 0
        X_total = row_df[v4_total_features]
        pred_total = total_model.predict(X_total)[0]
        
        # ML prediction
        for c in v4_ml_features:
            if c not in row_df.columns:
                row_df[c] = 0
        X_ml = row_df[v4_ml_features]
        
        fair_line = max(min(pred_margin, 25), -25)
        fair_total = max(min(pred_total, 70), 20)
        
        # --- SPREAD ---
        raw_away_spread = game['spread_line']
        if not pd.isna(raw_away_spread):
            home_spread = -1 * raw_away_spread
            vegas_home_margin = -1 * home_spread
            edge_spread = pred_margin - vegas_home_margin
            units_spread, conf_spread = calculate_kelly_units(abs(edge_spread))
            
            if units_spread > 0:
                pick_team = game['home_team'] if edge_spread > 0 else game['away_team']
                new_picks.append({
                    'week': WEEK_TO_BACKFILL,
                    'type': 'spread',
                    'home': game['home_team'],
                    'away': game['away_team'],
                    'pick_team': pick_team,
                    'pick': pick_team,
                    'line': f"{game['away_team']} {raw_away_spread:+}",
                    'fair_value': f"{game['away_team']} {fair_line:+.1f}",
                    'units': units_spread,
                    'confidence': conf_spread,
                    'game_id': game.get('game_id', ''),
                    'status': 'PENDING'
                })
        
        # --- TOTALS ---
        raw_total = game['total_line']
        if not pd.isna(raw_total):
            diff_total = fair_total - raw_total
            abs_total_edge = abs(diff_total)
            units_total, conf_total = calculate_totals_kelly(abs_total_edge)
            
            if units_total > 0:
                pick_type = "OVER" if diff_total > 0 else "UNDER"
                new_picks.append({
                    'week': WEEK_TO_BACKFILL,
                    'type': 'total',
                    'home': game['home_team'],
                    'away': game['away_team'],
                    'pick_team': pick_type,
                    'pick': pick_type,
                    'line': str(raw_total),
                    'fair_value': f"{fair_total:.1f}",
                    'units': units_total,
                    'confidence': conf_total,
                    'game_id': game.get('game_id', ''),
                    'status': 'PENDING'
                })
        
        # --- MONEYLINE ---
        if 'home_moneyline' in game and not pd.isna(game['home_moneyline']):
            prob_home_win = ml_model.predict_proba(X_ml)[0][1]
            prob_away_win = 1 - prob_home_win
            
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
            
            units_home, conf_home = calculate_moneyline_kelly(prob_home_win, v_prob_home, vegas_ml_home)
            units_away, conf_away = calculate_moneyline_kelly(prob_away_win, v_prob_away, vegas_ml_away)
            
            # Helper for odds
            def fmt_odds(o): return f"+{int(o)}" if o > 0 else f"{int(o)}"
            
            if units_home > 0 and units_home >= units_away:
                if prob_home_win >= 0.5:
                    fair_odds = int(-1 * (100 * prob_home_win) / (1 - prob_home_win)) if prob_home_win < 0.99 else -10000
                else:
                    fair_odds = int((100 * (1 - prob_home_win)) / prob_home_win) if prob_home_win > 0.01 else 10000
                
                new_picks.append({
                    'week': WEEK_TO_BACKFILL,
                    'type': 'moneyline',
                    'home': game['home_team'],
                    'away': game['away_team'],
                    'pick_team': f"{game['home_team']} ML",
                    'pick': f"{game['home_team']} ML",
                    'line': f"{game['home_team']} {int(vegas_ml_home):+}",
                    'fair_value': f"{game['home_team']} {fmt_odds(fair_odds)}",
                    'units': units_home,
                    'confidence': conf_home,
                    'game_id': game.get('game_id', ''),
                    'status': 'PENDING'
                })
            elif units_away > 0:
                if prob_away_win >= 0.5:
                    fair_odds = int(-1 * (100 * prob_away_win) / (1 - prob_away_win)) if prob_away_win < 0.99 else -10000
                else:
                    fair_odds = int((100 * (1 - prob_away_win)) / prob_away_win) if prob_away_win > 0.01 else 10000
                    
                new_picks.append({
                    'week': WEEK_TO_BACKFILL,
                    'type': 'moneyline',
                    'home': game['home_team'],
                    'away': game['away_team'],
                    'pick_team': f"{game['away_team']} ML",
                    'pick': f"{game['away_team']} ML",
                    'line': f"{game['away_team']} {int(vegas_ml_away):+}",
                    'fair_value': f"{game['away_team']} {fmt_odds(fair_odds)}",
                    'units': units_away,
                    'confidence': conf_away,
                    'game_id': game.get('game_id', ''),
                    'status': 'PENDING'
                })
    
    print(f"Generated {len(new_picks)} picks for Week {WEEK_TO_BACKFILL}")
    
    # Grade picks
    print("Grading picks against actual results...")
    schedule_wk = schedule[(schedule['season'] == CURRENT_SEASON) & (schedule['week'] == WEEK_TO_BACKFILL)]
    
    for pick in new_picks:
        game_row = schedule_wk[
            (schedule_wk['home_team'] == pick['home']) & 
            (schedule_wk['away_team'] == pick['away'])
        ]
        if not game_row.empty:
            result, profit = grade_pick(pick, game_row.iloc[0])
            pick['result'] = result
            pick['profit'] = profit
            pick['status'] = 'GRADED'
        else:
            pick['result'] = ''
            pick['profit'] = 0.0
    
    # Load existing history
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH)
    else:
        existing = pd.DataFrame()
    
    # Check for duplicates
    existing_wk18 = existing[existing['week'] == WEEK_TO_BACKFILL] if 'week' in existing.columns else pd.DataFrame()
    if len(existing_wk18) > 0:
        print(f"Warning: {len(existing_wk18)} Week {WEEK_TO_BACKFILL} picks already exist. Skipping to avoid duplicates.")
    else:
        # Append new picks
        new_df = pd.DataFrame(new_picks)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(new_picks)} new picks to {DATA_PATH}")
        
        # Summary
        wins = len([p for p in new_picks if p.get('result') == 'WIN'])
        losses = len([p for p in new_picks if p.get('result') == 'LOSS'])
        pushes = len([p for p in new_picks if p.get('result') == 'PUSH'])
        profit = sum([p.get('profit', 0) for p in new_picks])
        
        print(f"\n=== Week {WEEK_TO_BACKFILL} Summary ===")
        print(f"Record: {wins}-{losses}-{pushes}")
        print(f"Profit: {profit:+.2f} units")
        
        for p in new_picks:
            print(f"  [{p['type'].upper()}] {p['away']} @ {p['home']}: {p['pick']} ({p['units']}u) -> {p.get('result', 'N/A')} ({p.get('profit', 0):+.2f}u)")
