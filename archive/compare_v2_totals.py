
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import os

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_TOTAL_PATH = "models/v2_total.json"

# Features that make up the "New" model
NEW_FEATURES = [
    'points_scored', 'points_allowed',
    'total_ppg_L3', 'total_ppg_allowed_L3',
    'total_proj_score_L3', 'total_epa_proj',
    'home_points_scored_L3', 'away_points_scored_L3',
    'home_points_allowed_L3', 'away_points_allowed_L3',
    'home_points_scored', 'away_points_scored',
    'home_points_allowed', 'away_points_allowed',
    'diff_points_scored_L3', 'diff_points_allowed_L3' # Interaction diffs that might have been auto-generated
]

# Standard exclusions
IGNORE_COLS = [
    'season', 'week', 'game_id', 'result', 'total', 
    'home_score', 'away_score', 'spread_line', 'total_line',
    'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
    'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
    'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
    'home_su', 'away_su'
]

def train_and_eval(name, restricted_cols=None):
    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    df['total'] = df['home_score'] + df['away_score']
    train_df = df.dropna(subset=['total']).copy()
    
    # Feature Selection
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
    feature_cols = [c for c in feature_cols if train_df[c].dtype != object]
    
    if restricted_cols:
        feature_cols = [c for c in feature_cols if c not in restricted_cols]
        # Also remove rolling variations of restricted cols
        # e.g. if 'points_scored' is restricted, 'points_scored_L3' should be too if not explicitly listed
        # The list NEW_FEATURES includes specific L3s, but let's be safe
        final_cols = []
        for c in feature_cols:
            is_banned = False
            for banned in restricted_cols:
                if banned in c: 
                    is_banned = True
                    break
            if not is_banned:
                final_cols.append(c)
        feature_cols = final_cols

    print(f"\n--- Training {name} Model ---")
    print(f"Features: {len(feature_cols)}")
    
    X = train_df[feature_cols]
    y = train_df['total']
    
    # Model Config (Same for both)
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50,
        reg_alpha=0.1
    )
    
    last_season = train_df['season'].max()
    val_mask = train_df['season'] == last_season
    
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_val)
    mae = np.mean(np.abs(preds - y_val))
    rmse = np.sqrt(np.mean((preds - y_val)**2))
    
    print(f"Results for {name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return mae, rmse

if __name__ == "__main__":
    print("🔬 Running Model Comparison...")
    
    # 1. Baseline (Old Features only)
    # We pass the NEW_FEATURES list to be excluded
    base_mae, base_rmse = train_and_eval("BASELINE (Old)", restricted_cols=NEW_FEATURES)
    
    # 2. Enhanced (All Features)
    new_mae, new_rmse = train_and_eval("ENHANCED (New)", restricted_cols=[])
    
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"Baseline MAE: {base_mae:.4f}")
    print(f"Enhanced MAE: {new_mae:.4f}")
    
    diff = base_mae - new_mae
    pct = (diff / base_mae) * 100
    
    if diff > 0:
        print(f"✅ IMPROVEMENT: -{diff:.4f} MAE ({pct:.2f}%)")
    else:
        print(f"⚠️ REGRESSION: +{abs(diff):.4f} MAE")
