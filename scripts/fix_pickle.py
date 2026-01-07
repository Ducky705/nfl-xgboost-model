import pandas as pd
import pickle
import os

# Paths
DB_PATH = "data/nfl_db_v2.pkl"
FEATURES_PATH = "data/nfl_features_v2.pkl"

def fix_pickle():
    print(f"Loading {DB_PATH}...")
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
    
    if 'games_df' in db:
        print(f"Extracting games_df as features...")
        games_df = db['games_df']
        
        print(f"Saving to {FEATURES_PATH}...")
        with open(FEATURES_PATH, 'wb') as f:
            pickle.dump(games_df, f)
        print("Done.")
    else:
        print("Error: 'games_df' not found in db.")

if __name__ == "__main__":
    fix_pickle()
