"""
Protocol 705 - v2.5 Experiment Runner
Tests multiple improvements and compares to v2.3 baseline
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from itertools import product

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"

def run_experiment(X_train, y_train, X_val, y_val, params, name="Experiment"):
    """Train and evaluate with given params"""
    model = xgb.XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        n_jobs=-1,
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict
    preds = model.predict(X_val)
    
    # RMSE
    rmse = np.sqrt(((preds - y_val) ** 2).mean())
    
    # Betting Simulation
    val_df = X_val.copy()
    val_df['pred'] = preds
    val_df['actual'] = y_val.values
    val_df['spread_line'] = spread_lines_val  # Global from main
    
    wins = 0
    losses = 0
    
    for _, row in val_df.iterrows():
        if pd.isna(row['spread_line']): continue
        vegas_margin = -1 * row['spread_line']
        edge = row['pred'] - vegas_margin
        
        if abs(edge) > 1.5:  # Bet threshold
            # Bet Home if edge > 0
            if edge > 0:
                won = row['actual'] > vegas_margin
            else:
                won = row['actual'] < vegas_margin
            
            if won: wins += 1
            else: losses += 1
    
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    return {
        'name': name,
        'rmse': rmse,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'params': params
    }

if __name__ == "__main__":
    print("Loading Data...")
    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
    
    # Create Target
    df['result'] = df['home_score'] - df['away_score']
    
    # Split
    last_season = df['season'].max()
    train_df = df[df['season'] < last_season].dropna(subset=['result']).copy()
    val_df = df[df['season'] == last_season].dropna(subset=['result']).copy()
    
    # Identify Features
    IGNORE_COLS = [
        'season', 'week', 'game_id', 'result', 'total', 
        'home_score', 'away_score', 'spread_line', 'total_line',
        'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
        'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn'
    ]
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
    
    X_train = train_df[feature_cols]
    y_train = train_df['result']
    X_val = val_df[feature_cols]
    y_val = val_df['result']
    
    # Save spread lines for betting sim
    global spread_lines_val
    spread_lines_val = val_df['spread_line'].values
    
    print(f"Training: {len(X_train)} games, Validation: {len(X_val)} games")
    print(f"Features: {len(feature_cols)}")
    
    # --- EXPERIMENTS ---
    results = []
    
    # 1. Baseline (v2.3 params)
    print("\n[1/6] Baseline (v2.3 params)...")
    baseline_params = {
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:squarederror'
    }
    results.append(run_experiment(X_train, y_train, X_val, y_val, baseline_params, "Baseline (v2.3)"))
    
    # 2. Lower Learning Rate
    print("[2/6] Lower Learning Rate (0.005)...")
    lr_params = baseline_params.copy()
    lr_params['learning_rate'] = 0.005
    results.append(run_experiment(X_train, y_train, X_val, y_val, lr_params, "LR=0.005"))
    
    # 3. Higher Learning Rate
    print("[3/6] Higher Learning Rate (0.02)...")
    lr_params2 = baseline_params.copy()
    lr_params2['learning_rate'] = 0.02
    results.append(run_experiment(X_train, y_train, X_val, y_val, lr_params2, "LR=0.02"))
    
    # 4. Shallower Trees
    print("[4/6] Shallower Trees (depth=3)...")
    depth_params = baseline_params.copy()
    depth_params['max_depth'] = 3
    results.append(run_experiment(X_train, y_train, X_val, y_val, depth_params, "Depth=3"))
    
    # 5. Add Regularization
    print("[5/6] Add Regularization (L1+L2)...")
    reg_params = baseline_params.copy()
    reg_params['reg_alpha'] = 1.0  # L1
    reg_params['reg_lambda'] = 2.0  # L2
    results.append(run_experiment(X_train, y_train, X_val, y_val, reg_params, "Regularized"))
    
    # 6. Combo: Slower LR + Regularization
    print("[6/6] Combo (LR=0.005 + Regularization)...")
    combo_params = baseline_params.copy()
    combo_params['learning_rate'] = 0.005
    combo_params['reg_alpha'] = 0.5
    combo_params['reg_lambda'] = 1.0
    results.append(run_experiment(X_train, y_train, X_val, y_val, combo_params, "LR+Reg Combo"))
    
    # --- RESULTS ---
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_rate', ascending=False)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['name']}:")
        print(f"   RMSE: {row['rmse']:.2f}")
        print(f"   Record: {row['wins']}-{row['losses']}")
        print(f"   Win Rate: {row['win_rate']:.1f}%")
    
    print("\n" + "=" * 60)
    best = results_df.iloc[0]
    print(f"🏆 BEST: {best['name']} ({best['win_rate']:.1f}% Win Rate)")
    print("=" * 60)
    
    # Save results
    results_df.to_csv("data/hyperparameter_experiments.csv", index=False)
    print("\n✅ Results saved to data/hyperparameter_experiments.csv")
