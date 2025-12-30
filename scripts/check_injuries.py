
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
    
    if 'injury_stats' in db:
        print("\n--- Injury Stats ---")
        df = db['injury_stats']
        print(df.columns)
        print(df.head(2).to_dict('records'))
        
    print("\n--- Usage in Analysis ---")
    # Check simple lookup
    # usually injury_stats has [season, week, team, player, status, etc]
    
except Exception as e:
    print(f"Error: {e}")
