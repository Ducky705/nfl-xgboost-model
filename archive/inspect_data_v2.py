import pandas as pd
import pickle
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

try:
    with open('data/nfl_db_v2.pkl', 'rb') as f:
        db = pickle.load(f)
        schedule = db['schedule']
        
    CURRENT_SEASON = 2024 # Assumed based on file path v0.0.2 and context
    
    print(f"--- Week 18 Schedule Data (Season {CURRENT_SEASON}) ---")
    # Filter for Week 18 and current season
    week_18 = schedule[(schedule['week'] == 18) & (schedule['season'] == CURRENT_SEASON)]
    
    if week_18.empty:
        print("No Week 18 games found for this season.")
    else:
        # Check specifically for games where result is NaN (Upcoming)
        upcoming = week_18[week_18['result'].isna()]
        print(f"Total Week 18 Games: {len(week_18)}")
        print(f"Upcoming (Result is NaN): {len(upcoming)}")
        
        if not upcoming.empty:
            print("\n--- Upcoming Games Data ---")
            print(upcoming[['home_team', 'away_team', 'spread_line', 'total_line', 'result']])
            print("\nValue Counts for Spread Line:")
            print(upcoming['spread_line'].value_counts(dropna=False))
        else:
            print("All Week 18 games have results (graded).")
            print("\n--- Graded Games Data ---")
            print(week_18[['home_team', 'away_team', 'spread_line', 'total_line', 'result']].head())

except Exception as e:
    print(f"Error: {e}")
