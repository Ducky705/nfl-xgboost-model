import pandas as pd
import pickle
import os
import webbrowser
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate
from datetime import datetime
from scipy.stats import norm
import xgboost as xgb  # Required for DMatrix

import src.features as v2_features
import numpy as np
import shutil

# --- CONFIG ---
# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_STACK_PATH = "models/v3_ensemble_stack.pkl"
MODEL_TOTAL_PATH = "models/v3_total_stack.pkl" 
MODEL_ML_PATH = "models/v3_moneyline_stack.pkl"

from src.ensemble_model import VotingEnsemble


DATA_PATH = "data/betting_history.csv"
DOCS_PATH = "docs/index.html"
FIX_VEGAS_SIGNS = True 

# --- SYSTEM PARAMETERS ---
SYS_VAR_TOLERANCE = "LOW"
SYS_KELLY_SCALAR = 0.25
SYS_LIQUIDITY_REQ = "HIGH" 

# --- KELLY CRITERION FUNCTIONS ---
def calculate_kelly_units(abs_edge):
    """Calculates Kelly Unit size based on point spread edge.
    
    V3 - Manually Tuned for High ROI/Quality (User Pref).
    """
    STD_DEV = 12.53 
    PAYOUT_RATIO = 0.9091
    
    # V3 Manual Adjustments (Restoring 32-22-0 Sweet Spot)
    MIN_EDGE = 3.5    # Loosened from 3.6 to capture missed wins
    MAX_UNITS = 2.0
    KELLY_FRACTION = 0.035  # Increased slightly to boost ROI
    
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
    """Calculates Kelly Unit size for TOTALS.
    
    V3 - Manually Tuned for Positive ROI (Safety First).
    """
    STD_DEV = 13.5
    PAYOUT_RATIO = 0.9091
    
    # V3 Manual Adjustment (Emergency Fix for Negative ROI)
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
    """Calculates Kelly Unit size for MONEYLINE using true probability-based EV.
    
    V3 - Optuna-Optimized for ROI with min 15 bets.
    (Reverted to user-preferred High ROI settings 21-9-0)
    """
    # V3 Optuna-Optimized Parameters (ROI-focused, min 15 bets)
    MIN_EDGE = 0.0503  # 5.03% edge threshold
    MAX_UNITS = 2.0   # Cap
    MIN_ODDS = -134   # Optuna optimal
    MAX_ODDS = 127    # Optuna optimal (tighter range = higher quality)
    KELLY_FRACTION = 2.91  # Optuna optimal
    
    # 1. Odds Filter
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
    # V3.0: Optuna-Optimized Kelly
    units = round(min(max(0, kelly_fraction * KELLY_FRACTION), MAX_UNITS), 1)
    
    if units > 0:
        conf = "SOLID" if edge >= 0.15 else "LEAN"
    else:
        conf = "None"
        
    return units, conf


# --- SYSTEM CONFIDENCE FUNCTION (UNCHANGED) ---
def calculate_system_confidence(graded_df):
    """Calculates system confidence based on win rate variance across weeks."""
    if graded_df.empty or len(graded_df) < 5:
        return "CALIBRATING", "text-zinc-500", "Insufficient data"
    
    weekly_stats = graded_df.groupby('week').apply(
        lambda x: len(x[x['result'] == 'WIN']) / len(x) * 100 if len(x) > 0 else 0
    )
    
    if len(weekly_stats) < 3:
        return "CALIBRATING", "text-zinc-500", "Need more weeks"
    
    std_dev = weekly_stats.std()
    
    if std_dev < 10: return "STABLE", "text-acid-lime", f"σ={std_dev:.1f}%"
    elif std_dev < 20: return "MODERATE", "text-zinc-300", f"σ={std_dev:.1f}%"
    elif std_dev < 30: return "ELEVATED", "text-yellow-500", f"σ={std_dev:.1f}%"
    else: return "HIGH VOLATILITY", "text-warning-orange", f"σ={std_dev:.1f}%"



# --- SYSTEM CONFIDENCE FUNCTION ---
def calculate_system_confidence(graded_df):
    """Calculates system confidence based on win rate variance across weeks."""
    if graded_df.empty or len(graded_df) < 5:
        return "CALIBRATING", "text-zinc-500", "Insufficient data"
    
    # Group by week and calculate win rate per week
    weekly_stats = graded_df.groupby('week').apply(
        lambda x: len(x[x['result'] == 'WIN']) / len(x) * 100 if len(x) > 0 else 0
    )
    
    if len(weekly_stats) < 3:
        return "CALIBRATING", "text-zinc-500", "Need more weeks"
    
    # Calculate variance and standard deviation
    variance = weekly_stats.var()
    std_dev = weekly_stats.std()
    mean_wr = weekly_stats.mean()
    
    # Determine confidence level based on std deviation
    # Lower variance = more stable/confident system
    if std_dev < 10:
        return "STABLE", "text-acid-lime", f"σ={std_dev:.1f}%"
    elif std_dev < 20:
        return "MODERATE", "text-zinc-300", f"σ={std_dev:.1f}%"
    elif std_dev < 30:
        return "ELEVATED", "text-yellow-500", f"σ={std_dev:.1f}%"
    else:
        return "HIGH VOLATILITY", "text-warning-orange", f"σ={std_dev:.1f}%"

# --- CHART GENERATION FUNCTION ---
def generate_chart_data(df, n_bars=10):
    """
    Generates a list of percentage heights (20-100) for a bar chart
    based on the rolling cumulative profit of the last n_bars bets.
    If no history, returns a flat baseline.
    """
    if df.empty:
        return [{'h': 0, 'dir': 'up'}] * n_bars
        
    # Get all graded bets to calc proper cumsum
    graded_df = df[df['status'] == 'GRADED'].copy()
    
    if len(graded_df) < 2:
         return [{'h': 0, 'dir': 'up'}] * n_bars

    # Calculate cumulative profit on FULL history to preserve true level
    graded_df['cum_profit'] = graded_df['profit'].cumsum()
    
    # Now take the tail
    recent = graded_df.tail(n_bars)
    
    # Zero-Centered Normalization
    cum_profit = recent['cum_profit'].tolist()
    
    # Find the maximum absolute deviation from 0 to scale correctly
    abs_vals = [abs(x) for x in cum_profit]
    max_dev = max(abs_vals) if abs_vals else 1.0
    
    if max_dev == 0: max_dev = 1.0 
    
    chart_objs = []
    for val in cum_profit:
        pct_of_half = (abs(val) / max_dev) * 100
        pct_of_half = min(pct_of_half, 100.0)
        direction = 'up' if val >= 0 else 'down'
        chart_objs.append({'h': int(pct_of_half), 'dir': direction, 'val': val})
        
    # Pad with empty objects if needed
    while len(chart_objs) < n_bars:
        chart_objs.insert(0, {'h': 0, 'dir': 'up', 'val': 0})
        
    return chart_objs

# --- ORION TEMPLATE (SPREAD - CYAN/TERMINAL) ---

# --- 1. LOAD DB ---
if not os.path.exists(CACHE_PATH_V2):
    print("❌ Cache not found!")
    exit()

print(f"Loading V2 Database: {CACHE_PATH_V2}")
with open(CACHE_PATH_V2, 'rb') as f:
    db = pickle.load(f)

# Load Models
print(f"Loading Models...")
print(f"   Loading v3 Spread Ensemble: {MODEL_STACK_PATH}")
with open(MODEL_STACK_PATH, 'rb') as f:
    spread_model = pickle.load(f)

# Load v3 features
with open("models/v3_features.pkl", 'rb') as f:
    v4_features = pickle.load(f)

# v3 Total (Pulsar)
print(f"   Loading v3 Total Ensemble...")
with open("models/v3_total_stack.pkl", 'rb') as f:
    total_model = pickle.load(f)
with open("models/v3_total_features.pkl", 'rb') as f:
    v4_total_features = pickle.load(f)

# v3 Moneyline (Quasar)
print(f"   Loading v3 Moneyline Ensemble...")
with open("models/v3_moneyline_stack.pkl", 'rb') as f:
    ml_model = pickle.load(f)
with open("models/v3_moneyline_features.pkl", 'rb') as f:
    v4_ml_features = pickle.load(f)


# Load History (Features for context)
with open(FEATURES_PATH_V2, 'rb') as f:
    feature_df = pickle.load(f)

CURRENT_SEASON = feature_df['season'].max()

# Removed spread_to_moneyline function in favor of dedicated model

# --- 2. PREDICT UPCOMING ---
# --- 2. PREDICT UPCOMING ---
def run_predictions(db, spread_model, total_model, ml_model, override_week=None, min_week=0):
    schedule = db['schedule']
    base_stats = db['base_stats']
    injury_stats = db['injury_stats']

    # Filter schedule for upcoming games in CURRENT_SEASON
    if override_week:
        upcoming = schedule[(schedule['season'] == CURRENT_SEASON) & (schedule['week'] == override_week)].copy()
        next_week = override_week
    else:
        upcoming = schedule[(schedule['season'] == CURRENT_SEASON) & (schedule['result'].isna()) & (schedule['game_type'] == 'REG')].copy()
        # Enforce min_week constraint (strictly greater than last graded)
        upcoming = upcoming[upcoming['week'] > min_week]
        
        if upcoming.empty: 
            return [], 0
        next_week = upcoming['week'].min()

    print(f"Analyzing Week {next_week}...")
    
    print(f"Analyzing Week {next_week}...")
    
    # Run Pipeline
    # v3 refactor expects full db object (or at least dict with schedule etc)
    # We pass the full db object which contains 'schedule', 'base_stats', 'injury_stats', 'starters' etc.
    full_games = v2_features.engineering_pipeline(db)

    
    # Filter for next week
    week_df = full_games[(full_games['season'] == CURRENT_SEASON) & (full_games['week'] == next_week)].copy()
    
    preds = []
    
    # Get model features (all v3 now)
    spread_features = v4_features
    total_features = v4_total_features
    ml_features = v4_ml_features
    
    if not spread_features:
        print("Model feature names missing!")
        return [], 0

    for _, game in week_df.iterrows():
        # --- PREPARE DATA ---
        row_df = pd.DataFrame([game])
        
        # Spread Data (v4 Ensemble - uses DataFrame)
        for c in spread_features:
            if c not in row_df.columns: row_df[c] = 0
        X_spread = row_df[spread_features]
        
        # Total Data (v4 Ensemble - uses DataFrame)
        for c in total_features:
            if c not in row_df.columns: row_df[c] = 0
        X_total = row_df[total_features]

        # Moneyline Data (v4 Classifier Ensemble - uses DataFrame)
        for c in ml_features:
            if c not in row_df.columns: row_df[c] = 0
        X_ml = row_df[ml_features]
        
        # --- PREDICTIONS ---
        # --- PREDICTIONS ---
        # spread_model is now VotingEnsemble (supports .predict(X))
        pred_margin = spread_model.predict(X_spread)[0] # Home Margin
        pred_total = total_model.predict(X_total)[0]    # Total Score

        
        fair_line = max(min(pred_margin, 25), -25)
        fair_total = max(min(pred_total, 70), 20)
        
        # --- SPREAD BETTING ---
        # FIX: 'spread_line' in DB seems to be AWAY TEAM SPREAD (e.g. NE -13.5 means Away Team -13.5).
        # Standard convention: (-) is Favorite.
        raw_away_spread = game['spread_line']
        
        action_spread = "PASS"; units_spread = 0.0
        edge_spread = 0.0
        
        if not pd.isna(raw_away_spread):
            # If Away Spread is -13.5. Away is Favored.
            # Home Spread is +13.5.
            
            # Predict Margin (Home Score - Away Score)
            # spread_model predicts Home Margin.
            # If Home Margin is +10. (Home wins by 10).
            # If Home Spread is +13.5. Home covers easily.
            
            # Vegas Expectation for Home Margin: -1 * Home Spread.
            # NO. Vegas Expectation (Spread Line) usually means "Home Score + Spread = Away Score"? No.
            # Spread -7 (Home Fav). Home must win by 7.
            # Expected Margin = +7.
            # So Expected Margin = -1 * Home Spread.
            
            home_spread = -1 * raw_away_spread
            vegas_home_margin = -1 * home_spread # Expected Home Win Margin
            
            # Edge = Predicted Home Margin - Vegas Expected Home Margin
            # Example: Pred +10. Vegas +7 (Spread -7). Edge = +3.
            edge_spread = pred_margin - vegas_home_margin
            
            # Calculate Kelly
            units_spread, conf_spread = calculate_kelly_units(abs(edge_spread))
            
            # Determine Pick Team (Lean) regardless of units
            if edge_spread > 0:
                pick_team = game['home_team']
            else:
                pick_team = game['away_team']

            if units_spread > 0.0:
                action_spread = f"BET {pick_team} ({units_spread}u)"
            else:
                action_spread = "PASS"
                conf_spread = "None"
            
            # Display Logic
            display_line = f"{game['away_team']} {raw_away_spread:+}"
            
            clean_conf = conf_spread.replace("🔥 ", "").replace("⚠️ ", "").replace("💪 ", "").replace("None", "")
            
            preds.append({
                'Matchup': f"{game['away_team']} @ {game['home_team']}",
                'Vegas': f"{display_line}", 
                'Fair_Line': f"{game['away_team']} {fair_line:+.1f}",
                'Edge': round(abs(edge_spread), 1),
                'Action': action_spread,
                'Conf': conf_spread,
                'clean_conf': clean_conf,
                'units': units_spread,
                'pick': pick_team,
                'game_id': game.get('game_id', ''),
                'week': next_week,
                'type': 'spread',
                'home': game['home_team'],
                'away': game['away_team']
            })

        # --- TOTAL BETTING ---
        raw_total = game['total_line']
        if not pd.isna(raw_total):
            # Edge: Pred - Vegas
            # If Pred > Vegas (e.g. 50 > 45), Edge +5. Bet Over.
            # If Pred < Vegas (e.g. 40 < 45), Edge -5 (Absolute 5). Bet Under.
            
            diff_total = fair_total - raw_total
            abs_total_edge = abs(diff_total)
            
            # Use TOTALS-SPECIFIC Kelly with higher selectivity
            units_total, conf_total = calculate_totals_kelly(abs_total_edge)
            
            if units_total > 0.0:
                pick_type = "OVER" if diff_total > 0 else "UNDER"
                action_total = f"BET {pick_type} {raw_total}"
            else:
                pick_type = "OVER" if diff_total > 0 else "UNDER"
                action_total = "PASS"
                conf_total = "None"
                
            preds.append({
                'Matchup': f"{game['away_team']} @ {game['home_team']}",
                'Vegas': f"{raw_total}",
                'Fair_Line': f"{fair_total:.1f}",
                'Edge': round(abs_total_edge, 1),
                'Action': action_total,
                'Conf': conf_total,
                'clean_conf': conf_total.replace("🔥 ", "").replace("⚠️ ", "").replace("💪 ", "").replace("None", ""),
                'units': units_total,
                'pick': pick_type,
                'game_id': game.get('game_id', ''),
                'week': next_week,
                'type': 'total',
                'home': game['home_team'],
                'away': game['away_team']
            })

        # --- MONEYLINE BETTING (Evaluate BOTH Home and Away) ---
        if 'home_moneyline' in game and not pd.isna(game['home_moneyline']) and 'away_moneyline' in game and not pd.isna(game['away_moneyline']):
            # Predict Probabilities
            prob_home_win = ml_model.predict_proba(X_ml)[0][1]
            prob_away_win = 1 - prob_home_win
            
            vegas_ml_home = game['home_moneyline']
            vegas_ml_away = game['away_moneyline']
            
            # Vegas Implied Probabilities
            if vegas_ml_home < 0: v_prob_home = -vegas_ml_home / (-vegas_ml_home + 100)
            else: v_prob_home = 100 / (vegas_ml_home + 100)
            
            if vegas_ml_away < 0: v_prob_away = -vegas_ml_away / (-vegas_ml_away + 100)
            else: v_prob_away = 100 / (vegas_ml_away + 100)
            
            # Calculate edges for BOTH sides
            edge_home = prob_home_win - v_prob_home
            edge_away = prob_away_win - v_prob_away
            
            # Helper function for odds formatting
            def fmt_odds(o): return f"+{int(o)}" if o > 0 else f"{int(o)}"
            
            # Evaluate HOME team bet
            units_home, conf_home = calculate_moneyline_kelly(prob_home_win, v_prob_home, vegas_ml_home)
            
            # Evaluate AWAY team bet
            units_away, conf_away = calculate_moneyline_kelly(prob_away_win, v_prob_away, vegas_ml_away)
            
            # Pick the better opportunity (or pass if neither qualifies)
            if units_home > 0 and units_home >= units_away:
                # Bet HOME
                pick_team = game['home_team']
                pick_odds = vegas_ml_home
                pick_prob = prob_home_win
                pick_edge = edge_home
                pick_units = units_home
                pick_conf = conf_home
            elif units_away > 0:
                # Bet AWAY
                pick_team = game['away_team']
                pick_odds = vegas_ml_away
                pick_prob = prob_away_win
                pick_edge = edge_away
                pick_units = units_away
                pick_conf = conf_away
            else:
                # PASS - no value on either side
                pick_team = None
                pick_units = 0.0
                pick_conf = "None"
            
            # Convert model prob to fair odds for display
            if pick_team:
                if pick_prob >= 0.5:
                    fair_odds = int(-1 * (100 * pick_prob) / (1 - pick_prob)) if pick_prob < 0.99 else -10000
                else:
                    fair_odds = int((100 * (1 - pick_prob)) / pick_prob) if pick_prob > 0.01 else 10000
                
                action_ml = f"BET {pick_team} ML ({fmt_odds(pick_odds)})"
                # Include team name in Vegas and Fair strings
                vegas_str = f"{pick_team} {fmt_odds(pick_odds)}"
                fair_str = f"{pick_team} {fmt_odds(fair_odds)}"
                clean_conf_ml = pick_conf.replace("🔥 ", "").replace("⚠️ ", "").replace("💪 ", "").replace("None", "").lower()
            else:
                action_ml = "PASS"
                vegas_str = f"{game['home_team']} {fmt_odds(vegas_ml_home)}"
                # Calculate fair odds for home team even when passing
                if prob_home_win >= 0.5:
                    fair_home_odds = int(-1 * (100 * prob_home_win) / (1 - prob_home_win)) if prob_home_win < 0.99 else -10000
                else:
                    fair_home_odds = int((100 * (1 - prob_home_win)) / prob_home_win) if prob_home_win > 0.01 else 10000
                fair_str = f"{game['home_team']} {fmt_odds(fair_home_odds)}"
                pick_edge = max(abs(edge_home), abs(edge_away))
                clean_conf_ml = ""
                pick_team = game['home_team']  # Default for display
            
            preds.append({
                'Matchup': f"{game['away_team']} @ {game['home_team']}",
                'Vegas': vegas_str,
                'Fair_Line': fair_str,
                'Edge': round(abs(pick_edge) * 100, 1),
                'Action': action_ml,
                'Conf': pick_conf,
                'clean_conf': clean_conf_ml,
                'units': pick_units,
                'pick': f"{pick_team} ML",
                'game_id': game.get('game_id', ''),
                'week': next_week,
                'type': 'moneyline',
                'home': game['home_team'],
                'away': game['away_team']
            })
        
    return preds, next_week

# --- 3. OUTPUT ---
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    # Templates are now defined as variables (ORION, PULSAR, QUASAR) and loaded via from_string check render_page


    # --- DATA LOADING & PARTITIONING ---
    
    # Load History FIRST to determine min_week
    if os.path.exists(DATA_PATH):
        try:
            final_hist = pd.read_csv(DATA_PATH)
            # Determine last graded week
            graded_weeks = final_hist[final_hist['status'] == 'GRADED']['week']
            last_graded_week = graded_weeks.max() if not graded_weeks.empty else 0
        except:
            final_hist = pd.DataFrame()
            last_graded_week = 0
    else:
        final_hist = pd.DataFrame()
        last_graded_week = 0

    print(f"Last Graded Week: {last_graded_week}. Checking for pending games...")
    
    # Check if the last graded week is actually complete
    min_scan_week = last_graded_week
    pending_game_ids = set()
    
    if last_graded_week > 0:
        schedule = db['schedule']
        pending = schedule[
            (schedule['season'] == CURRENT_SEASON) & 
            (schedule['week'] == last_graded_week) & 
            (schedule['result'].isna()) &
            (schedule['game_type'] == 'REG')
        ]
        if not pending.empty:
            print(f"Week {last_graded_week} incomplete ({len(pending)} games pending). Staying on Week {last_graded_week}.")
            min_scan_week = last_graded_week - 1
            pending_game_ids = set(pending['game_id'])
            
            
    print(f"Scanning for Week {min_scan_week + 1}+")

    active_bets, next_week_num = run_predictions(db, spread_model, total_model, ml_model, min_week=min_scan_week)
    
    # Sort ALL Active Bets for Index/Selector logic
    if active_bets:
        # If we are restricting to pending games (because week is incomplete), filter here
        if pending_game_ids:
            print(f"Filtering Active Bets to {len(pending_game_ids)} pending games...")
            active_bets = [b for b in active_bets if b.get('game_id') in pending_game_ids]
            
        active_bets.sort(key=lambda x: (x['units'], x['Edge']), reverse=True)
    
    # Partition Active Bets
    spread_bets = [x for x in active_bets if x.get('type') == 'spread']
    total_bets = [x for x in active_bets if x.get('type') == 'total']
    ml_bets = [x for x in active_bets if x.get('type') == 'moneyline']

    # (History already loaded)
        
    env = Environment(loader=FileSystemLoader('templates'))
    
    # --- HELPER: RENDER PAGE ---
    def render_page(filename, title, model_name, bets_list, hist_df, template_filename):
        # Calc Stats
        graded = hist_df[hist_df['status'] == 'GRADED']
        wins = len(graded[graded['result'] == 'WIN'])
        losses = len(graded[graded['result'] == 'LOSS'])
        pushes = len(graded[graded['result'] == 'PUSH'])
        total_games = wins + losses + pushes
        profit = graded['profit'].sum()
        
        roi = round((profit/total_games)*100, 1) if total_games > 0 else 0.0
        win_pct = round((wins/total_games)*100, 1) if total_games > 0 else 0.0
        record_str = f"{wins}-{losses}-{pushes}"
        
        # Recent History - NOW SHOWS ALL GRADED PICKS (most recent first)
        recent = []
        if not graded.empty:
            # Sort by week descending, then by profit descending within each week
            recent_df = graded.sort_values(['week', 'profit'], ascending=[False, False])
            recent = recent_df.to_dict('records')
            
            # Add opponent field for each record
            for r in recent:
                try:
                    pick = str(r.get('pick_team', '')).strip()
                    home = str(r.get('home', '')).strip()
                    away = str(r.get('away', '')).strip()
                    
                    # Determine opponent
                    if 'ML' in pick:
                        pick_clean = pick.replace(' ML', '')
                        r['opponent'] = away if pick_clean == home else home
                    elif pick == 'OVER' or pick == 'UNDER':
                        r['opponent'] = ''  # Totals don't have opponent
                    else:
                        r['opponent'] = away if pick == home else home
                    
                    # Format week display
                    r['week_str'] = f"W{r.get('week', '')}"
                    
                except Exception:
                    r['opponent'] = ''
                    r['week_str'] = ''


        conf_level, conf_color, conf_detail = calculate_system_confidence(graded)
        
        # Determine Units Generated (Alpha)
        alpha_units = round(profit, 2)
        
        # Generate Chart Data
        chart_data = generate_chart_data(graded, n_bars=10)

        # USE SPECIFIC TEMPLATE
        template = env.get_template(template_filename)
        
        html_out = template.render(
            active_bets=bets_list,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            roi=roi,
            units=alpha_units,
            record=record_str,
            win_pct=win_pct,
            history=recent,
            chart_data=chart_data,
            confidence_level=conf_level,
            confidence_color=conf_color,
            confidence_detail=conf_detail,
            page_title=title,
            model_name=model_name,
            next_week=next_week_num, # Added next_week for Pulsar template
            system_var_tolerance=SYS_VAR_TOLERANCE,
            system_kelly_scalar=f"{SYS_KELLY_SCALAR}x",
            system_liquidity_req=SYS_LIQUIDITY_REQ
        )
        
        full_out_path = f"docs/{filename}"
        with open(full_out_path, "w", encoding="utf-8") as f: f.write(html_out)
        print(f"Generated {filename} ({len(bets_list)} active signals)")
        return roi, record_str, win_pct, alpha_units

    # --- PARTITION HISTORY ---
    if not final_hist.empty and 'type' in final_hist.columns:
        orion_hist = final_hist[final_hist['type'] == 'spread']
        pulsar_hist = final_hist[final_hist['type'] == 'total']
        quasar_hist = final_hist[final_hist['type'] == 'moneyline']
    else:
        # Fallback / Legacy behavior
        orion_hist = final_hist 
        pulsar_hist = pd.DataFrame(columns=final_hist.columns)
        quasar_hist = pd.DataFrame(columns=final_hist.columns)

    # --- GENERATE PAGES ---
    # 1. ORION (Spread)
    orion_roi, orion_rec, orion_pct, orion_alpha = render_page(
        "spread.html", 
        "ASTRALIS // ORION", 
        "ORION", 
        spread_bets, 
        orion_hist,
        "orion.html"
    )
    
    # 2. PULSAR (Totals)
    pulsar_roi, pulsar_rec, pulsar_pct, pulsar_alpha = render_page(
        "totals.html", 
        "ASTRALIS // PULSAR", 
        "PULSAR", 
        total_bets, 
        pulsar_hist,
        "pulsar.html"
    )
    
    # 3. QUASAR (Moneyline)
    quasar_roi, quasar_rec, quasar_pct, quasar_alpha = render_page(
        "moneyline.html", 
        "ASTRALIS // QUASAR", 
        "QUASAR", 
        ml_bets, 
        quasar_hist,
        "quasar.html"
    )
    
    # 4. INDEX (REMOVED - Replaced by Selector)
    # render_page("index.html"...) - DEPRECATED

    # --- SELECTOR GENERATION ---
    print("Generating Selector Page...")
    selector_template = env.get_template("selector.html")
    
    # Calculate Upcoming Alpha (Sum of units on active bets)
    upcoming_orion = sum([x['units'] for x in spread_bets])
    upcoming_pulsar = sum([x['units'] for x in total_bets])
    upcoming_quasar = sum([x['units'] for x in ml_bets])

    # Calculate Orion Stats for Selector
    orion_graded = orion_hist[orion_hist['status'] == 'GRADED']
    orion_wins = len(orion_graded[orion_graded['result'] == 'WIN'])
    orion_losses = len(orion_graded[orion_graded['result'] == 'LOSS'])
    orion_pushes = len(orion_graded[orion_graded['result'] == 'PUSH'])
    orion_total = orion_wins + orion_losses + orion_pushes
    orion_profit = orion_graded['profit'].sum()
    orion_roi = round((orion_profit/orion_total)*100, 1) if orion_total > 0 else 0.0
    
    # L30 Stats (Orion)
    orion_l30 = orion_graded.tail(30)
    orion_l30_wins = len(orion_l30[orion_l30['result'] == 'WIN'])
    orion_l30_losses = len(orion_l30[orion_l30['result'] == 'LOSS'])
    orion_l30_pushes = len(orion_l30[orion_l30['result'] == 'PUSH'])
    orion_l30_total = orion_l30_wins + orion_l30_losses + orion_l30_pushes
    orion_l30_record = f"{orion_l30_wins}-{orion_l30_losses}-{orion_l30_pushes}"
    orion_l30_pct = round((orion_l30_wins/orion_l30_total)*100, 1) if orion_l30_total > 0 else 0.0
    
    # Pulsar Stats (Totals)
    pulsar_graded = pulsar_hist[pulsar_hist['status'] == 'GRADED']
    pulsar_wins = len(pulsar_graded[pulsar_graded['result'] == 'WIN'])
    pulsar_losses = len(pulsar_graded[pulsar_graded['result'] == 'LOSS'])
    pulsar_pushes = len(pulsar_graded[pulsar_graded['result'] == 'PUSH'])
    pulsar_total = pulsar_wins + pulsar_losses + pulsar_pushes
    pulsar_profit = pulsar_graded['profit'].sum()
    pulsar_roi = round((pulsar_profit/pulsar_total)*100, 1) if pulsar_total > 0 else 0.0

    # L30 Stats (Pulsar)
    pulsar_l30 = pulsar_graded.tail(30)
    pulsar_l30_wins = len(pulsar_l30[pulsar_l30['result'] == 'WIN'])
    pulsar_l30_losses = len(pulsar_l30[pulsar_l30['result'] == 'LOSS'])
    pulsar_l30_pushes = len(pulsar_l30[pulsar_l30['result'] == 'PUSH'])
    pulsar_l30_total = pulsar_l30_wins + pulsar_l30_losses + pulsar_l30_pushes
    pulsar_l30_record = f"{pulsar_l30_wins}-{pulsar_l30_losses}-{pulsar_l30_pushes}"
    pulsar_l30_pct = round((pulsar_l30_wins/pulsar_l30_total)*100, 1) if pulsar_l30_total > 0 else 0.0
    
    # Quasar Stats (Moneyline)
    quasar_graded = quasar_hist[quasar_hist['status'] == 'GRADED']
    quasar_wins = len(quasar_graded[quasar_graded['result'] == 'WIN'])
    quasar_losses = len(quasar_graded[quasar_graded['result'] == 'LOSS'])
    quasar_pushes = len(quasar_graded[quasar_graded['result'] == 'PUSH'])
    quasar_total = quasar_wins + quasar_losses + quasar_pushes
    quasar_profit = quasar_graded['profit'].sum()
    quasar_roi = round((quasar_profit/quasar_total)*100, 1) if quasar_total > 0 else 0.0

    # L30 Stats (Quasar)
    quasar_l30 = quasar_graded.tail(30)
    quasar_l30_wins = len(quasar_l30[quasar_l30['result'] == 'WIN'])
    quasar_l30_losses = len(quasar_l30[quasar_l30['result'] == 'LOSS'])
    quasar_l30_pushes = len(quasar_l30[quasar_l30['result'] == 'PUSH'])
    quasar_l30_total = quasar_l30_wins + quasar_l30_losses + quasar_l30_pushes
    quasar_l30_record = f"{quasar_l30_wins}-{quasar_l30_losses}-{quasar_l30_pushes}"
    quasar_l30_pct = round((quasar_l30_wins/quasar_l30_total)*100, 1) if quasar_l30_total > 0 else 0.0
    
    selector_html = selector_template.render(
        next_week=next_week_num,
        season=CURRENT_SEASON,
        
        orion_alpha=round(upcoming_orion, 1),
        orion_record=orion_rec,
        orion_win_pct=orion_pct,
        orion_roi=orion_roi,
        orion_l30_record=orion_l30_record,
        orion_l30_pct=orion_l30_pct,
        
        pulsar_alpha=round(upcoming_pulsar, 1),
        pulsar_record=pulsar_rec,
        pulsar_win_pct=pulsar_pct,
        pulsar_roi=pulsar_roi,
        pulsar_l30_record=pulsar_l30_record,
        pulsar_l30_pct=pulsar_l30_pct,
        
        quasar_alpha=round(upcoming_quasar, 1),
        quasar_record=quasar_rec,
        quasar_win_pct=quasar_pct,
        quasar_roi=quasar_roi,
        quasar_l30_record=quasar_l30_record,
        quasar_l30_pct=quasar_l30_pct
    )
    
    SELECTOR_PATH = "docs/selector.html"
    with open(SELECTOR_PATH, "w", encoding="utf-8") as f: f.write(selector_html)
    print(f"Selector Page Updated: {os.path.abspath(SELECTOR_PATH)}")

    # --- ASSETS ---
    try:
        shutil.copy("data/betting_history.csv", "docs/ledger.csv")
        print("Ledger copied to docs/ledger.csv")
    except Exception as e:
        print(f"Could not copy ledger: {e}")

    webbrowser.open(f"file://{os.path.abspath(SELECTOR_PATH)}")
