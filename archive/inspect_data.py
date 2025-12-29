import pandas as pd
import pickle

try:
    with open('data/nfl_db_v2.pkl', 'rb') as f:
        db = pickle.load(f)
        schedule = db['schedule']
        
    print("--- Week 18 Schedule Data ---")
    week_18 = schedule[schedule['week'] == 18]
    if week_18.empty:
        print("No Week 18 games found.")
    else:
        print(week_18[['home_team', 'away_team', 'spread_line', 'total_line', 'result']].head(10))
        print("\nColumn types:")
        print(week_18[['spread_line', 'total_line']].dtypes)
except Exception as e:
    print(f"Error: {e}")
