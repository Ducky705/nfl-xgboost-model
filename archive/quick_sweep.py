"""Quick analysis of spread thresholds"""
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from scipy.stats import norm
import v2_features

with open("data/nfl_db_v2.pkl", 'rb') as f:
    db = pickle.load(f)

schedule = db['schedule']
full_games = v2_features.engineering_pipeline(schedule, db['base_stats'], db['injury_stats'])

spread_model = xgb.Booster()
spread_model.load_model("models/v2_spread.json")
spread_features = spread_model.feature_names

results = []
for min_edge in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    for kelly in [0.10, 0.15, 0.20]:
        profit = 0.0
        bets = 0
        wins = 0
        
        week_games = full_games[(full_games['season']==2025) & (full_games['result'].notna())]
        
        for _, game in week_games.iterrows():
            real = schedule[schedule['game_id']==game['game_id']].iloc[0]
            if pd.isna(real['spread_line']): continue
            
            row_df = pd.DataFrame([game])
            for c in spread_features:
                if c not in row_df.columns: row_df[c] = 0
            pred = spread_model.predict(xgb.DMatrix(row_df[spread_features]))[0]
            
            vegas = real['spread_line'] * -1
            edge = abs(pred - (-vegas))
            
            if edge < min_edge: continue
            
            z = edge / 12.53
            p = norm.cdf(z)
            kelly_pct = (p - (1-p)/0.9091) * 100
            units = round(min(max(0, kelly_pct * kelly), 5.0), 1)
            
            if units > 0:
                bets += 1
                actual = real['home_score'] - real['away_score']
                pick_home = (pred - (-vegas)) > 0
                
                if pick_home:
                    won = actual > (-vegas)
                else:
                    won = actual < (-vegas)
                    
                if won:
                    profit += units * 0.9091
                    wins += 1
                else:
                    profit -= units
        
        wr = wins/bets*100 if bets > 0 else 0
        results.append((min_edge, kelly, bets, wins, bets-wins, round(wr,1), round(profit,2)))

print("\nSPREAD THRESHOLD ANALYSIS:")
print("="*70)
print(f"{'MIN_EDGE':>8} {'KELLY':>6} {'BETS':>5} {'W-L':>8} {'WR%':>6} {'PROFIT':>10}")
print("="*70)
for r in results:
    print(f"{r[0]:>8.1f} {r[1]:>6.2f} {r[2]:>5} {r[3]:>3}-{r[4]:<4} {r[5]:>5.1f}% {r[6]:>+10.2f}u")

print("\n" + "="*70)
print("PROFITABLE SETTINGS (sorted by bets):")
print("="*70)
profitable = [r for r in results if r[6] > 0]
profitable.sort(key=lambda x: x[2], reverse=True)
for r in profitable[:5]:
    print(f"MIN_EDGE={r[0]:.1f}, KELLY={r[1]:.2f} -> {r[2]} bets, {r[5]:.1f}% WR, {r[6]:+.2f}u")
