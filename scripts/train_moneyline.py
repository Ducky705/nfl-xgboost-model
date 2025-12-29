
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import os
from sklearn.metrics import log_loss, accuracy_score

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
PRUNED_FEATURES_PATH = "data/pruned_features_moneyline.txt"
MODEL_ML_PATH = "models/v2_moneyline.json"

# Features to exclude from training (identifiers, labels)
IGNORE_COLS = [
    'season', 'week', 'game_id', 'result', 'total', 
    'home_score', 'away_score', 'spread_line', 'total_line',
    'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
    'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
    'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
    'home_su', 'away_su',
    # Keep interaction terms and sums, they are useful for ML too.
]

def train_moneyline_model():
    print(f"Loading features from {FEATURES_PATH_V2}...")
    if not os.path.exists(FEATURES_PATH_V2):
        print("Feature file not found. Please run v2_features.py first.")
        return

    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
        
    # Create Targets
    # Check if home_su exists or create it
    if 'home_su' not in df.columns:
        df['home_su'] = (df['home_score'] > df['away_score']).astype(int)
        
    # Handle Ties for Classification? 
    # Ties are < 0.5% of games. Dropping them for binary classification training is cleanest.
    df['is_tie'] = (df['home_score'] == df['away_score'])
    
    # Filter completed games (Must have result)
    train_df = df.dropna(subset=['home_score', 'away_score']).copy()
    train_df = train_df[~train_df['is_tie']] # Drop ties
    
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

    print(f"Training Moneyline (Win/Loss) Model on {len(feature_cols)} features...")
    
    X = train_df[feature_cols]
    y = train_df['home_su']
    
    # --- OPTIMIZED PARAMS (Dec 2025) ---
    # Best LogLoss: 0.6442
    params = {
        'n_estimators': 2271, 
        'learning_rate': 0.001747, 
        'max_depth': 3, 
        'subsample': 0.6437, 
        'colsample_bytree': 0.5168, 
        'reg_alpha': 3.269, 
        'reg_lambda': 3.687, 
        'min_child_weight': 10,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    ml_model = xgb.XGBClassifier(**params)
    
    # Validation Split
    val_season = 2025
    print(f"Validating on Season {val_season}")
    
    val_mask = train_df['season'] == val_season
    
    X_train = X[~val_mask]
    y_train = y[~val_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    if len(X_val) > 0:
        ml_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        # Evaluate
        probs = ml_model.predict_proba(X_val)[:, 1] # Probability of Class 1 (Home Win)
        preds = (probs > 0.5).astype(int)
        
        loss = log_loss(y_val, probs)
        acc = accuracy_score(y_val, preds)
        
        print(f"Validation Log Loss: {loss:.4f} (Lower is better)")
        print(f"Validation Accuracy: {acc*100:.2f}%")
    else:
        print("No validation data for 2025, training on full set.")
        ml_model.fit(X, y, verbose=100)
    
    ml_model.feature_names = feature_cols # Save names
    ml_model.save_model(MODEL_ML_PATH)
    print(f"✅ Moneyline Model saved to {MODEL_ML_PATH}")
    

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_moneyline_model()
