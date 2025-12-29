
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import os

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
PRUNED_FEATURES_PATH = "data/pruned_features_spread.txt"
MODEL_SPREAD_PATH = "models/v2_spread.json"

# Features to exclude from training (identifiers, labels)
IGNORE_COLS = [
    'season', 'week', 'game_id', 'result', 'total', 
    'home_score', 'away_score', 'spread_line', 'total_line',
    'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
    'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
    'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
    'home_su', 'away_su',
]

def train_spread_model():
    print(f"Loading features from {FEATURES_PATH_V2}...")
    if not os.path.exists(FEATURES_PATH_V2):
        print("Feature file not found. Please run v2_features.py first.")
        return

    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    # Create Targets
    df['result'] = df['home_score'] - df['away_score'] # Home Margin
    
    # Filter completed games
    train_df = df.dropna(subset=['result']).copy()
    
    # Identify Feature Columns
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
    feature_cols = [c for c in feature_cols if train_df[c].dtype != object] # Safety

    # --- FEATURE PRUNING ---
    if os.path.exists(PRUNED_FEATURES_PATH):
        print(f"Applying Feature Pruning from {PRUNED_FEATURES_PATH}...")
        with open(PRUNED_FEATURES_PATH, 'r') as f:
            keep_features = set(line.strip() for line in f.readlines())
        
        original_count = len(feature_cols)
        feature_cols = [c for c in feature_cols if c in keep_features]
        print(f"   Pruned features: {original_count} -> {len(feature_cols)}")
    else:
        print("   No pruned feature list found, using all features.")

    print(f"Training Spread Model on {len(feature_cols)} features...")
    
    X = train_df[feature_cols]
    y_spread = train_df['result']
    
    # --- OPTIMIZED PARAMS (Dec 2025) ---
    # Best MAE: 9.8231
    params = {
        'n_estimators': 1096, 
        'learning_rate': 0.003224, 
        'max_depth': 2, 
        'subsample': 0.6717, 
        'colsample_bytree': 0.9276, 
        'reg_alpha': 2.513, 
        'reg_lambda': 3.546, 
        'min_child_weight': 6,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }
    
    spread_model = xgb.XGBRegressor(**params)
    
    # Validation Split (Walk-Forward Style for Reporting)
    # Using 2025 as validation since it's the current incomplete season
    val_season = 2025
    print(f"Validating on Season {val_season}")
    
    val_mask = train_df['season'] == val_season
    
    X_train = X[~val_mask]
    y_train = y_spread[~val_mask]
    X_val = X[val_mask]
    y_val = y_spread[val_mask]
    
    if len(X_val) > 0:
        spread_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        # Evaluate
        preds = spread_model.predict(X_val)
        mae = np.mean(np.abs(preds - y_val))
        print(f"Validation MAE: {mae:.2f} points")
    else:
        print("No validation data for 2025, training on full set.")
        spread_model.fit(X, y_spread, verbose=100)

    
    spread_model.feature_names = feature_cols # Save names
    spread_model.save_model(MODEL_SPREAD_PATH)
    print(f"✅ Spread Model saved to {MODEL_SPREAD_PATH}")
    

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_spread_model()
