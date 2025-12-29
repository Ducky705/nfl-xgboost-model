
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_SPREAD_PATH = "models/v2_spread.json"
MODEL_TOTAL_PATH = "models/v2_total.json"

# Features to exclude from training (identifiers, labels)
IGNORE_COLS = [
    'season', 'week', 'game_id', 'result', 'total', 
    'home_score', 'away_score', 'spread_line', 'total_line',
    'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
    'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn'
]

def train_models():
    print(f"Loading features from {FEATURES_PATH_V2}...")
    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    # Create Targets
    df['result'] = df['home_score'] - df['away_score']
    df['total'] = df['home_score'] + df['away_score']
    
    # Filter completed games
    train_df = df.dropna(subset=['result', 'total']).copy()
    
    # Identify Feature Columns
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
    print(f"Training on {len(feature_cols)} features.")
    
    X = train_df[feature_cols]
    y_spread = train_df['result']
    y_total = train_df['total']
    
    # --- SPREAD MODEL ---
    print("\nTraining Spread Model (Home Margin)...")
    spread_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Use a time-based split for validation (Last season in dataset)
    last_season = train_df['season'].max()
    val_mask = train_df['season'] == last_season
    
    X_train = X[~val_mask]
    y_train = y_spread[~val_mask]
    X_val = X[val_mask]
    y_val = y_spread[val_mask]
    
    spread_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    spread_model.save_model(MODEL_SPREAD_PATH)
    print(f"✅ Spread Model saved to {MODEL_SPREAD_PATH}")
    
    # --- TOTAL MODEL ---
    print("\nTraining Total Model (Combined Score)...")
    total_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    y_train_tot = y_total[~val_mask]
    y_val_tot = y_total[val_mask]
    
    total_model.fit(
        X_train, y_train_tot,
        eval_set=[(X_val, y_val_tot)],
        verbose=100
    )
    
    total_model.save_model(MODEL_TOTAL_PATH)
    print(f"✅ Total Model saved to {MODEL_TOTAL_PATH}")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_models()
