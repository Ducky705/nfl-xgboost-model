import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import sys

# Try to import optuna, handle missing dependency
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("❌ Optuna not found. Please install it with: pip install optuna")
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, log_loss

# --- CONFIG ---
FEATURES_PATH = "data/nfl_features_v2.pkl"
PRUNED_FEATURES_PATH = "data/pruned_features.txt"
RESULTS_PATH = "models/tuning_results.txt"

# --- FEATURES TO IGNORE ---
IGNORE_COLS = [
    'season', 'week', 'game_id', 'result', 'total', 
    'home_score', 'away_score', 'spread_line', 'total_line',
    'gameday', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline',
    'div_game', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
    'home_cover', 'away_cover', 'ats_win', 'su_win', 'over_hit',
    'home_su', 'away_su', 'is_tie'
]

def load_data():
    print(f"Loading Data from {FEATURES_PATH}...")
    with open(FEATURES_PATH, 'rb') as f:
        df = pickle.load(f)
    
    # Ensure Targets
    if 'result' not in df.columns:
        df['result'] = df['home_score'] - df['away_score']
    if 'total' not in df.columns:
        df['total'] = df['home_score'] + df['away_score']
    if 'home_su' not in df.columns:
        df['home_su'] = (df['home_score'] > df['away_score']).astype(int)
        
    return df

def get_feature_cols(df, use_pruning=True):
    # Base numeric features
    cols = [c for c in df.columns if c not in IGNORE_COLS and df[c].dtype in [np.float64, np.float32, int, float]]
    
    # Filtering Logic
    if use_pruning and os.path.exists(PRUNED_FEATURES_PATH):
        print(f"   Applying pruning from {PRUNED_FEATURES_PATH}...")
        with open(PRUNED_FEATURES_PATH, 'r') as f:
            keep_features = set(line.strip() for line in f.readlines())
        
        # Intersection
        pruned_cols = [c for c in cols if c in keep_features]
        print(f"   Reduced features from {len(cols)} to {len(pruned_cols)}")
        return pruned_cols
    else:
        print(f"   Using all {len(cols)} features (No pruning file found)")
        return cols

# --- WALK FORWARD VALIDATION ---
def walk_forward_cv(model_class, params, df, feature_cols, target_col, is_classifier=False):
    """
    Performs Walk-Forward Validation on 2024 and 2025 seasons.
    """
    errors = []
    
    # 1. Fold 1: Train <= 2023, Test 2024
    train_1 = df[df['season'] <= 2023]
    test_1 = df[df['season'] == 2024]
    
    if not test_1.empty:
        model_1 = model_class(**params)
        model_1.fit(train_1[feature_cols], train_1[target_col])
        preds_1 = model_1.predict_proba(test_1[feature_cols])[:,1] if is_classifier else model_1.predict(test_1[feature_cols])
        
        if is_classifier:
            score = log_loss(test_1[target_col], preds_1)
        else:
            score = mean_absolute_error(test_1[target_col], preds_1)
        errors.append(score)
        
    # 2. Fold 2: Train <= 2024, Test 2025
    train_2 = df[df['season'] <= 2024]
    test_2 = df[df['season'] == 2025] # Current season
    
    if not test_2.empty:
        model_2 = model_class(**params)
        model_2.fit(train_2[feature_cols], train_2[target_col])
        preds_2 = model_2.predict_proba(test_2[feature_cols])[:,1] if is_classifier else model_2.predict(test_2[feature_cols])
        
        if is_classifier:
            score = log_loss(test_2[target_col], preds_2)
        else:
            score = mean_absolute_error(test_2[target_col], preds_2)
        errors.append(score)
        
    return np.mean(errors) if errors else float('inf')

# --- OBJECTIVES ---
def objective_spread(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }
    return walk_forward_cv(xgb.XGBRegressor, params, df_global, features_global, 'result', is_classifier=False)

def objective_total(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }
    return walk_forward_cv(xgb.XGBRegressor, params, df_global, features_global, 'total', is_classifier=False)

def objective_moneyline(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    return walk_forward_cv(xgb.XGBClassifier, params, df_global, features_global, 'home_su', is_classifier=True)

# Global vars for optimization
df_global = None
features_global = None

def run_tuning(target_type="spread"):
    global df_global, features_global
    
    df = load_data()
    # Filter valid lines
    if target_type == 'spread':
        df = df.dropna(subset=['result'])
    elif target_type == 'total':
        df = df.dropna(subset=['total'])
    
    df_global = df
    features_global = get_feature_cols(df, use_pruning=True)
    
    print(f"\n🚀 Starting Optimization for {target_type.upper()} Model...")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    
    if target_type == 'spread':
        study.optimize(objective_spread, n_trials=30) # 30 trials for speed, increase for production
    elif target_type == 'total':
        study.optimize(objective_total, n_trials=30)
    elif target_type == 'moneyline':
        study.optimize(objective_moneyline, n_trials=30)
        
    print(f"\n✅ Optimization Complete for {target_type}")
    print(f"   Best MAE: {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")
    
    # Save Results
    with open(RESULTS_PATH, 'a') as f:
        f.write(f"\n--- {target_type.upper()} MODEL ---\n")
        f.write(f"Best MAE: {study.best_value:.4f}\n")
        f.write(f"Params: {study.best_params}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=['spread', 'total', 'moneyline'], help='Target model to tune')
    args = parser.parse_args()
    
    run_tuning(args.target)
