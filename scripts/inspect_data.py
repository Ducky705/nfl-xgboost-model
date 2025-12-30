
import pickle
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd()))

try:
    # Load DB
    print("Loading DB...")
    with open("data/nfl_db_v2.pkl", "rb") as f:
        db = pickle.load(f)
    
    print("\n--- DB Keys ---")
    print(db.keys())
    
    if 'injury_stats' in db:
        print("\n--- Injury Stats Sample ---")
        print(db['injury_stats'].head())
        print(db['injury_stats'].columns)
        
    if 'starters' in db:
        print("\n--- Starters Sample ---")
        print(db['starters'].head())
        
    # Load Model
    print("\nLoading Spread Model...")
    with open("models/v3_ensemble_stack.pkl", "rb") as f:
        model = pickle.load(f)
        
    print("\n--- Model Type ---")
    print(type(model))
    
    if hasattr(model, 'estimators'):
        print(f"Estimators: {len(model.estimators)}")
        for name, est in model.estimators:
            print(f"  - {name}: {type(est)}")
            if hasattr(est, 'feature_importances_'):
                print(f"    Has feature_importances_")

except Exception as e:
    print(f"Error: {e}")
