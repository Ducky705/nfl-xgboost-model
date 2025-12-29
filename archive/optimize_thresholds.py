import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import sys
import scipy.stats as stats

# Suppress Optuna logging to clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
FEATURES_PATH = "data/nfl_features_v2.pkl"
MODEL_SPREAD = "models/v2_spread.json"
MODEL_TOTAL = "models/v2_total.json"
MODEL_ML = "models/v2_moneyline.json"

PRUNED_SPREAD = "data/pruned_features_spread.txt"
PRUNED_TOTAL = "data/pruned_features_totals.txt"
PRUNED_ML = "data/pruned_features_moneyline.txt"

def load_data():
    print("Loading Data...")
    with open(FEATURES_PATH, 'rb') as f:
        df = pickle.load(f)
    
    # Ensure targets exist
    if 'result' not in df.columns:
        df['result'] = df['home_score'] - df['away_score']
    if 'total' not in df.columns:
        df['total'] = df['home_score'] + df['away_score']
    if 'home_su' not in df.columns:
        df['home_su'] = (df['home_score'] > df['away_score']).astype(int)
        
    return df

def get_feature_cols(all_cols, pruned_path):
    if not os.path.exists(pruned_path):
        return [c for c in all_cols if c not in ['season', 'week', 'label', 'target']] # Fallback
    
    with open(pruned_path, 'r') as f:
        keep_features = set(line.strip() for line in f.readlines())
        
    return [c for c in all_cols if c in keep_features]

def generate_predictions(df):
    print("Generating Base Predictions...")
    
    # Spread
    if os.path.exists(MODEL_SPREAD):
        model = xgb.XGBRegressor()
        model.load_model(MODEL_SPREAD)
        cols = get_feature_cols(df.columns, PRUNED_SPREAD)
        if cols:
            df['pred_spread_margin'] = model.predict(df[cols])
        else:
            df['pred_spread_margin'] = 0
            
    # Total
    if os.path.exists(MODEL_TOTAL):
        model = xgb.XGBRegressor()
        model.load_model(MODEL_TOTAL)
        cols = get_feature_cols(df.columns, PRUNED_TOTAL)
        if cols:
            df['pred_total'] = model.predict(df[cols])
        else:
            df['pred_total'] = 0
            
    # Moneyline
    if os.path.exists(MODEL_ML):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_ML)
        cols = get_feature_cols(df.columns, PRUNED_ML)
        if cols:
            df['prob_home_win'] = model.predict_proba(df[cols])[:, 1]
        else:
            df['prob_home_win'] = 0.5
            
    return df

# --- OPTIMIZATION LOGIC ---

def kelly_criterion(odds, prob, fraction):
    """
    Standard Kelly Formula: f* = (bp - q) / b
    where b = net odds (decimal - 1), p = probability, q = 1-p
    """
    if prob <= 0 or prob >= 1: return 0
    b = odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    return max(0, f_star * fraction)

def optimize_spread(df):
    print("\n--- Optimizing SPREAD Thresholds ---")
    
    # Filter to graded games
    valid = df.dropna(subset=['result', 'spread_line']).copy()
    # Focus on 2024-2025 or just recent
    valid = valid[valid['season'] >= 2024]
    
    def objective(trial):
        min_edge = trial.suggest_float('min_edge', 0.5, 6.0)
        max_units = trial.suggest_float('max_units', 1.0, 5.0)
        kelly_frac = trial.suggest_float('kelly_fraction', 0.05, 0.25)
        
        balance = 0
        wagered = 0
        
        for _, row in valid.iterrows():
            # Logic: pred_margin vs line
            # If pred_margin (Home - Away) > -Spread (Vegas Home Margin)
            # E.g. Pred +7, Line -3   (7 > 3). Edge = 4.
            # Vegas Spread is Home Team. -3.5 means Home Favored by 3.5.
            # Compare: Pred Margin (H-A) vs -1 * Spread
            
            vegas_margin = -1 * row['spread_line']
            pred_margin = row['pred_spread_margin']
            
            edge = pred_margin - vegas_margin 
            
            # Bet Home
            if edge > min_edge:
                payout = 0.9091 # -110
                # Simple confidence based sizing or fixed?
                # Using Kelly-ish adaptation: 
                # Improve: Use generic Kelly with implied prob?
                # For spread, assume 50% implied, adjusted by edge?
                # Let's use simple unit scaling for now as per original code logic but parameterized
                
                # Original logic: units = base * confidence
                # Let's use proper Kelly using perceived probability
                # Perceived Win Prob?
                # Approx: 50% + (Edge * 0.02) ? 
                # Standard deviation of NFL spread is ~13-14.
                # Z-score = Edge / 13.5
                # Win Prob = Norm.cdf(Z)
                # import scipy.stats as stats
                z = edge / 13.5
                my_prob = stats.norm.cdf(z + 0.03) # +0.03 small drift/vig adjustment? No.
                
                # Assume -110 odds -> 1.9091 decimal
                decimal_odds = 1.9091
                f = kelly_criterion(decimal_odds, my_prob, kelly_frac)
                units = min(f * 100, max_units) # f * 100 to map to "Units" if bankroll=100
                
                # Grade
                if units > 0:
                    wagered += units
                    # Did Home Cover?
                    actual_margin = row['result']
                    if actual_margin > vegas_margin:
                        balance += units * payout
                    elif actual_margin < vegas_margin:
                        balance -= units
                    # Push = 0
            
            # Bet Away
            elif edge < -min_edge:
                # Away Edge
                abs_edge = -edge
                z = abs_edge / 13.5
                my_prob = stats.norm.cdf(z)
                
                decimal_odds = 1.9091
                f = kelly_criterion(decimal_odds, my_prob, kelly_frac)
                units = min(f * 100, max_units)
                
                if units > 0:
                    wagered += units
                    # Did Away Cover?
                    actual_margin = row['result']
                    if actual_margin < vegas_margin: # Away wins/covers
                        balance += units * payout
                    elif actual_margin > vegas_margin:
                        balance -= units

        # OBJECTIVE: Maximize Total Profit (Units), not ROI.
        # This naturally incentivizes volume if it's profitable.
        if wagered == 0:
            return 0.0 # No profit
            
        return balance

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100) # Increased trials
    
    print("Best Spread Params:", study.best_params)
    print("Best Spread Profit:", study.best_value)
    return study.best_params

def optimize_total(df):
    print("\n--- Optimizing TOTAL Thresholds (Max Profit) ---")
    valid = df.dropna(subset=['total', 'total_line']).copy()
    valid = valid[valid['season'] >= 2024]
    
    def objective(trial):
        # Wider search range for Balanced discovery
        min_edge = trial.suggest_float('min_edge', 0.5, 7.0)
        max_units = trial.suggest_float('max_units', 1.0, 5.0)
        kelly_frac = trial.suggest_float('kelly_fraction', 0.01, 0.25)
        
        balance = 0
        wagered = 0
        
        for _, row in valid.iterrows():
            pred = row['pred_total']
            line = row['total_line']
            diff = pred - line
            
            # OVER
            if diff > min_edge:
                z = diff / 13.5
                my_prob = stats.norm.cdf(z)
                f = kelly_criterion(1.9091, my_prob, kelly_frac)
                units = min(f * 100, max_units)
                if units > 0:
                    wagered += units
                    if row['total'] > line:
                        balance += units * 0.9091
                    elif row['total'] < line:
                        balance -= units
            
            # UNDER
            elif diff < -min_edge:
                abs_diff = -diff
                z = abs_diff / 13.5
                my_prob = stats.norm.cdf(z)
                f = kelly_criterion(1.9091, my_prob, kelly_frac)
                units = min(f * 100, max_units)
                if units > 0:
                    wagered += units
                    if row['total'] < line:
                        balance += units * 0.9091
                    elif row['total'] > line:
                        balance -= units
                        
        return balance

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("Best Total Params:", study.best_params)
    print("Best Total Profit:", study.best_value)
    return study.best_params

def optimize_moneyline(df):
    print("\n--- Optimizing MONEYLINE Thresholds (Max Profit) ---")
    valid = df.dropna(subset=['home_su', 'home_moneyline', 'away_moneyline']).copy()
    valid = valid[valid['season'] >= 2024]
    
    def get_implied_prob(ml):
        if ml < 0: return (-ml) / (-ml + 100)
        return 100 / (ml + 100)
    
    def get_decimal_odds(ml):
        if ml > 0: return (ml / 100) + 1
        return (100 / -ml) + 1

    def objective(trial):
        # Search for lower edges?
        min_edge = trial.suggest_float('min_edge', 0.00, 0.20) # Allow 0% edge check
        max_units = trial.suggest_float('max_units', 0.5, 3.0)
        kelly_frac = trial.suggest_float('kelly_fraction', 0.01, 0.15)
        
        balance = 0
        wagered = 0
        
        for _, row in valid.iterrows():
            prob_home = row['prob_home_win']
            prob_away = 1 - prob_home
            
            # Home Bet
            implied_home = get_implied_prob(row['home_moneyline'])
            edge_home = prob_home - implied_home
            
            if edge_home > min_edge:
                odds_dec = get_decimal_odds(row['home_moneyline'])
                f = kelly_criterion(odds_dec, prob_home, kelly_frac)
                units = min(f * 100, max_units)
                
                if units > 0:
                    wagered += units
                    if row['home_su'] == 1:
                        balance += units * (odds_dec - 1)
                    else:
                        balance -= units
            
            # Away Bet
            implied_away = get_implied_prob(row['away_moneyline'])
            edge_away = prob_away - implied_away
            
            if edge_away > min_edge:
                odds_dec = get_decimal_odds(row['away_moneyline'])
                f = kelly_criterion(odds_dec, prob_away, kelly_frac)
                units = min(f * 100, max_units)
                
                if units > 0:
                    wagered += units
                    if row['home_su'] == 0: # Away Win
                        balance += units * (odds_dec - 1)
                    else:
                        balance -= units
                        
        return balance

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print("Best Moneyline Params:", study.best_params)
    print("Best Moneyline Profit:", study.best_value)
    return study.best_params

if __name__ == "__main__":
    df = load_data()
    df = generate_predictions(df)
    
    spread_params = optimize_spread(df)
    total_params = optimize_total(df)
    ml_params = optimize_moneyline(df)
    
    print("\n\n====== OPTIMIZATION COMPLETE ======")
    print("SPREAD:", spread_params)
    print("TOTAL:", total_params)
    print("MONEYLINE:", ml_params)
    print("====================================")
