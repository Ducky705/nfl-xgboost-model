import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ensemble_model import VotingClassifierEnsemble

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_STACK_PATH = "models/v4_moneyline_stack.pkl"

def train_ensemble():
    print("Training v4 Quasar (Moneyline) Ensemble...")
    
    if not os.path.exists(FEATURES_PATH_V2):
        print("Features not found. Run features.py first.")
        return

    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    print(f"   Loaded features: {df.shape}")
    
    # Filter completed games
    train_df = df.dropna(subset=['home_score', 'away_score']).copy()
    
    # Target: Home Win (1 = Home Win, 0 = Away Win/Tie)
    train_df['home_win'] = (train_df['home_score'] > train_df['away_score']).astype(int)
    y = train_df['home_win']
    
    # Feature Selection
    ignore_cols = [
        'season', 'week', 'result', 'home_score', 'away_score', 
        'spread_line', 'total_line', 'home_team', 'away_team', 'game_id',
        'home_moneyline', 'away_moneyline', 'gameday', 'gametime',
        'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
        'home_su', 'away_su', 'market_result', 'old_game_id', 'home_win'
    ]
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]
    
    print(f"   Training on {len(feature_cols)} numeric features.")
    
    X = train_df[feature_cols]
    
    # --- MODEL DEFS ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1
    )
    
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=6,
        verbose=0,
        allow_writing_files=False
    )
    
    # --- ENSEMBLE ---
    ensemble = VotingClassifierEnsemble(
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
        accuracy = np.mean(preds == y_val)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Also check probability calibration
        probas = ensemble.predict_proba(X_val)[:, 1]
        from sklearn.metrics import log_loss
        logloss = log_loss(y_val, probas)
        print(f"Validation Log Loss: {logloss:.4f}")
    else:
        print("   No validation data, training on full set.")
        ensemble.fit(X, y)
        
    # --- SAVE ---
    os.makedirs("models", exist_ok=True)
    with open(MODEL_STACK_PATH, 'wb') as f:
        pickle.dump(ensemble, f)
        
    with open("models/v4_moneyline_features.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
        
    print(f"Moneyline Ensemble Model Saved to {MODEL_STACK_PATH}")

if __name__ == "__main__":
    train_ensemble()
