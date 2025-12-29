import pandas as pd
import pickle

pd.set_option('display.max_rows', 30)
pd.set_option('display.width', 150)

with open('data/nfl_db_v2.pkl', 'rb') as f:
    db = pickle.load(f)
    schedule = db['schedule']

week18 = schedule[(schedule['week'] == 18) & (schedule['season'] == 2025)]
print(f"Week 18 Games: {len(week18)}")
print(week18[['home_team', 'away_team', 'spread_line', 'total_line', 'result']].to_string())
