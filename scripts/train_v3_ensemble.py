
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ensemble_model import VotingEnsemble

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_STACK_PATH = "models/v4_ensemble_stack.pkl"
DATA_CACHE_PATH = "data/nfl_db_v2.pkl"

def train_ensemble():
    print("Training v4 Quasar Ensemble...")
    
    if not os.path.exists(FEATURES_PATH_V2):
        print("Features not found. Run features.py first.")
        return

    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    # --- PREPROCESSING ---
    print(f"   Loaded features: {df.shape}")
    
    # Filter completed games
    train_df = df.dropna(subset=['result']).copy()
    
    # Target
    y = train_df['result']
    
    # Feature Selection (Exclude labels)
    ignore_cols = [
        'season', 'week', 'result', 'home_score', 'away_score', 
        'spread_line', 'total_line', 'home_team', 'away_team', 'game_id',
        'home_moneyline', 'away_moneyline', 'gameday', 'gametime',
        'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
        'home_su', 'away_su', 'market_result', 'old_game_id'
    ]
    
    # Also ignore string columns (CatBoost can handle them, but let's stick to numeric for shared input)
    # Actually, CatBoost loves categorical features.
    # But for an ensemble with XGB/LGBM, we usually encoding them.
    # Let's drop non-numeric for now to be safe with all 3.
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    print(f"   Training on {len(feature_cols)} numeric features.")
    
    X = train_df[feature_cols]
    
    # --- MODEL DEFS ---
    # 1. XGBoost (The Veteran)
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    # 2. LightGBM (The Speedster)
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1
    )
    
    # 3. CatBoost (The Specialist) - Use default or light tuning
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        verbose=0,
        allow_writing_files=False
    )
    
    # --- ENSEMBLE ---
    # Weighted Average: Maybe favor XGBoost slightly as it's proven?
    # Let's try equal weights first.
    
    ensemble = VotingEnsemble(
        estimators=[
            ('xgb', xgb_model),
            ('lgbm', lgbm_model), # LightGBM needs plain names?
            ('cat', cat_model)
        ],
        weights=[0.4, 0.3, 0.3] # Slight bias to XGB
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
        
        # Breakdown by Model to check contribution?
        # (Optional, simpler to just trust the ensemble for now)
    else:
        print("   No validation data, training on full set.")
        ensemble.fit(X, y)
        
    # --- SAVE ---
    # We save the Custom Ensemble class pickle
    # Note: This requires the class definition to be available when loading.
    # Ideally we'd save individual models, but pickle is fine for this repo.
    
    os.makedirs("models", exist_ok=True)
    with open(MODEL_STACK_PATH, 'wb') as f:
        pickle.dump(ensemble, f)
        
    # Also save feature names list
    with open("models/v4_features.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
        
    print(f"Ensemble Model Saved to {MODEL_STACK_PATH}")

if __name__ == "__main__":
    train_ensemble()
