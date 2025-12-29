"""
V3.2 Parameter Sweep - Find optimal spread threshold for balanced volume/profit.
"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from scipy.stats import norm
import v2_features

CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
MODEL_SPREAD_PATH = "models/v2_spread.json"

def calculate_kelly_units(abs_edge, min_edge, kelly_frac=0.25, max_units=5.0):
    STD_DEV = 12.53
    PAYOUT_RATIO = 0.9091
    
    if abs_edge < min_edge:
        return 0.0
    
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score)
    q = 1.0 - p
    
    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * kelly_frac)
    return round(min(kelly_units, max_units), 1)

def simulate_spread_with_threshold(full_games, schedule, min_edge, kelly_frac):
    """Simulate spread betting with given threshold."""
    spread_model = xgb.Booster()
    spread_model.load_model(MODEL_SPREAD_PATH)
    spread_features = spread_model.feature_names
    
    current_season = 2025
    weeks = sorted(full_games[(full_games['season'] == current_season) & (full_games['result'].notna())]['week'].unique())
    
    total_profit = 0.0
    bet_count = 0
    wins = 0
    losses = 0
    
    for week in weeks:
        week_games = full_games[(full_games['season'] == current_season) & (full_games['week'] == week)].copy()
        
        for _, game in week_games.iterrows():
            real_res = schedule[(schedule['game_id'] == game['game_id'])].iloc[0]
            
            if pd.isna(real_res['home_score']) or pd.isna(real_res['away_score']):
                continue
                
            home_score = real_res['home_score']
            away_score = real_res['away_score']
            
            # Prediction
            row_df = pd.DataFrame([game])
            for c in spread_features:
                if c not in row_df.columns: row_df[c] = 0
            X_spread = xgb.DMatrix(row_df[spread_features])
            pred_margin = spread_model.predict(X_spread)[0]
            
            # Spread logic
            raw_away_spread = real_res['spread_line']
            if pd.isna(raw_away_spread):
                continue
                
            home_spread = -1 * raw_away_spread
            vegas_home_margin = -1 * home_spread
            edge_spread = pred_margin - vegas_home_margin
            
            units = calculate_kelly_units(abs(edge_spread), min_edge, kelly_frac)
            
            if units > 0:
                bet_count += 1
                pick_home = edge_spread > 0
                
                actual_margin = home_score - away_score
                
                if pick_home:
                    if actual_margin > vegas_home_margin:
                        total_profit += units * 0.9091
                        wins += 1
                    elif actual_margin < vegas_home_margin:
                        total_profit -= units
                        losses += 1
                else:
                    if actual_margin < vegas_home_margin:
                        total_profit += units * 0.9091
                        wins += 1
                    elif actual_margin > vegas_home_margin:
                        total_profit -= units
                        losses += 1
    
    return {
        'min_edge': min_edge,
        'kelly_frac': kelly_frac,
        'bets': bet_count,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
        'profit': round(total_profit, 2)
    }

if __name__ == "__main__":
    print("Loading data...")
    with open(CACHE_PATH_V2, 'rb') as f:
        db = pickle.load(f)
    
    schedule = db['schedule']
    base_stats = db['base_stats']
    injury_stats = db['injury_stats']
    
    print("Running feature engineering...")
    full_games = v2_features.engineering_pipeline(schedule, base_stats, injury_stats)
    
    print("\n" + "="*70)
    print("SPREAD THRESHOLD SWEEP - Finding V3.2 Optimal Settings")
    print("="*70)
    
    # Test range of MIN_EDGE values
    results = []
    for min_edge in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for kelly in [0.10, 0.15, 0.20, 0.25]:
            res = simulate_spread_with_threshold(full_games, schedule, min_edge, kelly)
            results.append(res)
            print(f"MIN_EDGE={min_edge:.1f}, KELLY={kelly:.2f} | {res['bets']:3d} bets | "
                  f"{res['wins']}W-{res['losses']}L ({res['win_rate']:.1f}%) | "
                  f"Profit: {res['profit']:+.2f}u")
    
    print("\n" + "="*70)
    print("TOP 5 BY PROFIT (positive only):")
    print("="*70)
    profitable = [r for r in results if r['profit'] > 0]
    profitable.sort(key=lambda x: x['profit'], reverse=True)
    for r in profitable[:5]:
        print(f"MIN_EDGE={r['min_edge']:.1f}, KELLY={r['kelly_frac']:.2f} | "
              f"{r['bets']} bets | {r['win_rate']:.1f}% WR | {r['profit']:+.2f}u")
    
    print("\n" + "="*70)
    print("BEST BALANCED (most bets while still profitable):")
    print("="*70)
    profitable.sort(key=lambda x: x['bets'], reverse=True)
    for r in profitable[:3]:
        print(f"MIN_EDGE={r['min_edge']:.1f}, KELLY={r['kelly_frac']:.2f} | "
              f"{r['bets']} bets | {r['win_rate']:.1f}% WR | {r['profit']:+.2f}u")
