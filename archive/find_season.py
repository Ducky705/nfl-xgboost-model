import pandas as pd
import pickle

pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

try:
    with open('data/nfl_db_v2.pkl', 'rb') as f:
        db = pickle.load(f)
        schedule = db['schedule']
        
    print("Searching for Week 18 games matching user's list (ARI @ LA, GB @ MIN)...")
    
    for season in [2023, 2024, 2025]:
        rows = schedule[(schedule['week'] == 18) & (schedule['season'] == season)]
        if not rows.empty:
            print(f"\n--- Season {season} Week 18 ---")
            print(rows[['home_team', 'away_team', 'spread_line', 'total_line', 'result']].head(20))
            
            # Check for specific match
            ari_la = rows[(rows['away_team'] == 'ARI') & (rows['home_team'] == 'LA')]
            gb_min = rows[(rows['away_team'] == 'GB') & (rows['home_team'] == 'MIN')]
            
            if not ari_la.empty:
                print(f"FOUND ARI @ LA in Season {season}!")
            if not gb_min.empty:
                print(f"FOUND GB @ MIN in Season {season}!")

except Exception as e:
    print(f"Error: {e}")
