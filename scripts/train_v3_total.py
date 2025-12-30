import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ensemble_model import VotingEnsemble

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_STACK_PATH = "models/v4_total_stack.pkl"

def train_ensemble():
    print("Training v4 Pulsar (Totals) Ensemble...")
    
    if not os.path.exists(FEATURES_PATH_V2):
        print("Features not found. Run features.py first.")
        return

    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    print(f"   Loaded features: {df.shape}")
    
    # Filter completed games with total data
    train_df = df.dropna(subset=['home_score', 'away_score']).copy()
    
    # Target: Total Points (Home + Away)
    train_df['total_points'] = train_df['home_score'] + train_df['away_score']
    y = train_df['total_points']
    
    # Feature Selection
    ignore_cols = [
        'season', 'week', 'result', 'home_score', 'away_score', 
        'spread_line', 'total_line', 'home_team', 'away_team', 'game_id',
        'home_moneyline', 'away_moneyline', 'gameday', 'gametime',
        'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
        'home_su', 'away_su', 'market_result', 'old_game_id', 'total_points'
    ]
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    print(f"   Training on {len(feature_cols)} numeric features.")
    
    X = train_df[feature_cols]
    
    # --- MODEL DEFS ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        verbose=-1
    )
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        verbose=0,
        allow_writing_files=False
    )
    
    # --- ENSEMBLE ---
    ensemble = VotingEnsemble(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgbm_model),
            ('cat', cat_model)
        ],
        weights=[0.4, 0.3, 0.3]
    )
    
    # Validation
    val_season = 2025
    val_mask = train_df['season'] == val_season
    
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    if len(X_val) > 0:
        print(f"   Validating on Season {val_season} ({len(X_val)} games)...")
        ensemble.fit(X_train, y_train)
        
        preds = ensemble.predict(X_val)
        mae = np.mean(np.abs(preds - y_val))
        print(f"Validation MAE: {mae:.4f}")
    else:
        print("   No validation data, training on full set.")
        ensemble.fit(X, y)
        
    # --- SAVE ---
    os.makedirs("models", exist_ok=True)
    with open(MODEL_STACK_PATH, 'wb') as f:
        pickle.dump(ensemble, f)
        
    with open("models/v4_total_features.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
        
    print(f"Total Ensemble Model Saved to {MODEL_STACK_PATH}")

if __name__ == "__main__":
    train_ensemble()
