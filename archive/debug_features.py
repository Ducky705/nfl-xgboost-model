
import pickle
import pandas as pd

try:
    with open("data/nfl_db_v2.pkl", 'rb') as f:
        data = pickle.load(f)
        
    stats = data['base_stats']
    print("Stats columns:", stats.columns.tolist())
    print("Duplicate columns:", stats.columns[stats.columns.duplicated()])
    
    # Try running the rolling logic on a small subset
    df = stats.sort_values(['team', 'season', 'week']).copy()
    col = 'off_epa'
    shifted = df.groupby(['season', 'team'])[col].shift(1)
    print("Shifted type:", type(shifted))
    print("Shifted shape:", shifted.shape)
    
    print("Detailed Transform Test:")
    try:
        res = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.expanding().mean())
        print("Transform success! Shape:", res.shape)
    except Exception as e:
        print("Transform failed:", e)
        
except Exception as e:
    print("Load failed:", e)
