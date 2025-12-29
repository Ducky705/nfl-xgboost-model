
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_SPREAD_PATH = "models/v2_spread.json"
MODEL_TOTAL_PATH = "models/v2_total.json"

def evaluate():
    print("Loading Data & Models...")
    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    # Load as Boosters
    spread_model = xgb.Booster()
    spread_model.load_model(MODEL_SPREAD_PATH)
    
    total_model = xgb.Booster()
    total_model.load_model(MODEL_TOTAL_PATH)
    
    # Prepare Data
    df['result'] = df['home_score'] - df['away_score']
    df['total_score'] = df['home_score'] + df['away_score']
    
    # Evaluation Set
    last_season = df['season'].max()
    eval_df = df[df['season'] == last_season].dropna(subset=['result', 'total_score']).copy()
    
    print(f"\nEvaluating on Season {last_season} ({len(eval_df)} games)...")
    
    # --- FEATURES ---
    # Smart Feature alignment
    # Get feature names from the model
    # Note: feature_names might be just f0, f1... if not saved with names. 
    # But usually save_model from sklearn wrapper preserves names in JSON.
    model_features = spread_model.feature_names
    
    if not model_features:
        print("⚠️ Warning: Model has no feature names. Using all numeric features (risky alignment).")
        # Fallback to logic
        IGNORE_COLS = [
            'season', 'week', 'game_id', 'result', 'total', 'total_score',
            'home_score', 'away_score', 'spread_line', 'total_line',
            'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
            'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn'
        ]
        feature_cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
        X = eval_df[feature_cols]
        dtest = xgb.DMatrix(X)
    else:
        # Filter eval_df to match model features exactly
        print(f"Aligning to {len(model_features)} model features...")
        # Ensure all columns exist (fill 0 if missing - unexpected but safe)
        for c in model_features:
            if c not in eval_df.columns:
                eval_df[c] = 0
        X = eval_df[model_features]
        dtest = xgb.DMatrix(X)
    
    # --- PREDICT ---
    eval_df['pred_spread'] = spread_model.predict(dtest)
    eval_df['pred_total'] = total_model.predict(dtest)
    
    # --- METRICS ---
    eval_df['spread_error'] = eval_df['pred_spread'] - eval_df['result']
    rmse_spread = np.sqrt((eval_df['spread_error'] ** 2).mean())
    mae_spread = eval_df['spread_error'].abs().mean()
    
    eval_df['total_error'] = eval_df['pred_total'] - eval_df['total_score']
    rmse_total = np.sqrt((eval_df['total_error'] ** 2).mean())
    mae_total = eval_df['total_error'].abs().mean()
    
    # --- METRICS ---
    eval_df['spread_error'] = eval_df['pred_spread'] - eval_df['result']
    rmse_spread = np.sqrt((eval_df['spread_error'] ** 2).mean())
    mae_spread = eval_df['spread_error'].abs().mean()
    
    eval_df['total_error'] = eval_df['pred_total'] - eval_df['total_score']
    rmse_total = np.sqrt((eval_df['total_error'] ** 2).mean())
    mae_total = eval_df['total_error'].abs().mean()
    
    print(f"\nPerformance Metrics (Season {last_season}):")
    print(f"   Spread RMSE: {rmse_spread:.2f} (Target: <13.5)")
    print(f"   Spread MAE:  {mae_spread:.2f}")
    print(f"   Total RMSE:  {rmse_total:.2f}")
    print(f"   Total MAE:   {mae_total:.2f}")

    # --- BETTING SIMULATION (Simple) ---
    # Bet if Edge > 1.5
    print("\nSimple Betting Simulation (Edge > 1.5):")
    
    # Spread Bets
    eval_df['vegas_spread'] = -1 * eval_df['spread_line'] # Convert to Home Margin 
    # Note: spread_line usually: Home -3.5 -> spread_line=-3.5. So Home Margin required = +3.5. 
    # Wait, nfl_data_py spread_line is Home - Away?
    # No, spread_line is "Points favored by". -3.5 means Home is favored by 3.5.
    # Result is Home - Away. So if Home wins 24-20, Result=4.
    # If Spread is -3.5, Home covers.
    # If Pred Result is 5.0 (Home wins by 5), and Vegas says Home wins by 3.5 (Spread -3.5).
    # Edge is 1.5 points of value on Home.
    
    # Let's align signs:
    # Vegas Line (Home Margin) = -1 * spread_line?
    # nfl_data_py: spread_line is positive if away favored? No.
    # Usually: spread_line = -3.5 (Home Favored).
    # So "Vegas Expectation of Home Margin" = +3.5.
    # So `vegas_margin = -1 * spread_line`.
    
    eval_df['vegas_margin'] = -1 * eval_df['spread_line']
    eval_df['edge_spread'] = eval_df['pred_spread'] - eval_df['vegas_margin']
    
    # Bet Home if Pred > Vegas + 1.5
    # Bet Away if Pred < Vegas - 1.5
    bets = []
    for _, row in eval_df.iterrows():
        if pd.isna(row['vegas_margin']): continue
        
        # SPREAD
        if row['edge_spread'] > 1.5:
            # Bet Home
            # Win if Result > Vegas Margin
            won = row['result'] > row['vegas_margin']
            push = row['result'] == row['vegas_margin']
            bets.append({'type': 'Spread', 'result': 'WIN' if won else ('PUSH' if push else 'LOSS')})
        elif row['edge_spread'] < -1.5:
            # Bet Away
            # Win if Result < Vegas Margin
            won = row['result'] < row['vegas_margin']
            push = row['result'] == row['vegas_margin']
            bets.append({'type': 'Spread', 'result': 'WIN' if won else ('PUSH' if push else 'LOSS')})
            
    bet_df = pd.DataFrame(bets)
    if not bet_df.empty:
        print("\n   Spread Bets:")
        vc = bet_df['result'].value_counts()
        wins = vc.get('WIN', 0)
        losses = vc.get('LOSS', 0)
        pushes = vc.get('PUSH', 0)
        total = wins + losses
        wr = (wins/total)*100 if total > 0 else 0
        print(f"   Record: {wins}-{losses}-{pushes}")
        print(f"   Win Rate: {wr:.1f}%")
        
    else:
        print("   No Spread bets triggered.")

    # Feature Importance (Top 10)
    print("\nTop 10 Predictive Features (Spread):")
    imp = spread_model.get_score(importance_type='gain')
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in sorted_imp:
        print(f"   {k}: {v:.1f}")

if __name__ == "__main__":
    evaluate()
