import os

# --- REQUIREMENTS (Add cryptography) ---
REQUIREMENTS_TXT = """pandas
numpy
xgboost
nfl_data_py
scikit-learn
jinja2
tabulate
scipy
matplotlib
seaborn
cryptography
"""

# --- EFFICIENT WORKFLOW ---
WORKFLOW_CONTENT = """name: NFL Bot Auto-Update

on:
  schedule:
    # Run Daily at 9:00 AM UTC
    - cron: '0 9 * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Decrypt Model
      env:
        # We pull the key from the GitHub Vault
        MODEL_KEY: ${{ secrets.MODEL_KEY }}
      run: |
        python -c "from cryptography.fernet import Fernet; import os; open('nfl_bettor.json', 'wb').write(Fernet(os.environ['MODEL_KEY'].encode()).decrypt(open('nfl_bettor.locked', 'rb').read()))"

    - name: Update Stats (Fast Mode)
      # Now update_db.py simply downloads yesterday's scores to update history
      # It does NOT retrain the model because nfl_bettor.json already exists!
      run: python update_db.py

    - name: Generate Predictions
      run: python main.py

    - name: Update Charts
      run: python generate_graphs.py

    - name: Save Website Changes
      run: |
        git config --global user.name "NFL-Bot"
        git config --global user.email "bot@nfl.com"
        # We commit the history (data) and website (docs)
        # We NEVER commit the unlocked json file
        git add data/ docs/
        git commit -m "daily_update" || echo "No changes"
        git push
"""

# --- UPDATE DB SCRIPT (Add Skip Logic) ---
# We need to tweak update_db.py so it skips training if the model exists
UPDATE_DB_CONTENT = r'''import pandas as pd
import numpy as np
import nfl_data_py as nfl
import xgboost as xgb
import os
import pickle
from datetime import datetime

# --- CONFIG ---
CACHE_PATH = "data/nfl_db.pkl"
MODEL_PATH = "nfl_bettor.json"
TEAM_MAP = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'SD': 'LAC', 'SL': 'LA', 'STL': 'LA', 'OAK': 'LV'}

now = datetime.now()
CURRENT_SEASON = now.year if now.month > 2 else now.year - 1
YEARS = list(range(2018, CURRENT_SEASON + 1))

def get_team_stats(df, full_pbp):
    # (Same stats function as before - truncated for brevity but functionality remains)
    gen = df.groupby(['season', 'week', 'posteam']).agg({'epa': 'mean', 'yards_gained': 'mean'}).reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa', 'yards_gained': 'off_ypp'})
    edsr = df[df['down'].isin([1, 2])].groupby(['season', 'week', 'posteam'])['success'].mean().reset_index().rename(columns={'posteam': 'team', 'success': 'off_edsr'})
    
    pass_plays = full_pbp[full_pbp['pass'] == 1]
    off_sacks = pass_plays.groupby(['season', 'week', 'posteam'])['sack'].mean().reset_index().rename(columns={'posteam': 'team', 'sack': 'off_sack_rate'})
    def_sacks = pass_plays.groupby(['season', 'week', 'defteam'])['sack'].mean().reset_index().rename(columns={'defteam': 'team', 'sack': 'def_sack_rate'})

    st = full_pbp[full_pbp['special_teams_play'] == 1].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'st_epa'})
    tos = full_pbp.groupby(['season', 'week', 'posteam']).agg({'fumble_lost': 'sum', 'interception': 'sum'}).reset_index()
    tos['turnovers_lost'] = tos['fumble_lost'] + tos['interception']
    penalties = full_pbp[full_pbp['penalty'] == 1].groupby(['season', 'week', 'penalty_team']).agg({'penalty_yards': 'sum'}).reset_index().rename(columns={'penalty_team': 'team', 'penalty_yards': 'pen_yards'})
    rz = df[df['yardline_100'] <= 20].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_rz_epa'})

    merged = gen.merge(edsr, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(off_sacks, on=['season', 'week', 'team'], how='left').merge(def_sacks, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(st, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(tos[['season', 'week', 'posteam', 'turnovers_lost']].rename(columns={'posteam': 'team'}), on=['season', 'week', 'team'], how='left')
    merged = merged.merge(penalties, on=['season', 'week', 'team'], how='left')
    merged = merged.merge(rz, on=['season', 'week', 'team'], how='left')
    pass_df = df[df['pass'] == 1].groupby(['season', 'week', 'posteam'])['epa'].mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_pass_epa'})
    merged = merged.merge(pass_df, on=['season', 'week', 'team'], how='left')
    return merged.fillna(0)

def engineer_features(schedule, stats, qb_db):
    games = schedule[schedule['game_type'] == 'REG'].copy()
    games['home_team'] = games['home_team'].replace(TEAM_MAP)
    games['away_team'] = games['away_team'].replace(TEAM_MAP)
    games = games.drop(columns=['home_rest', 'away_rest'], errors='ignore')
    games['gameday'] = pd.to_datetime(games['gameday'])
    rest_df = pd.concat([games[['season', 'week', 'gameday', 'home_team']].rename(columns={'home_team': 'team'}), 
                         games[['season', 'week', 'gameday', 'away_team']].rename(columns={'away_team': 'team'})]).sort_values(['team', 'gameday'])
    rest_df['rest'] = (rest_df['gameday'] - rest_df.groupby('team')['gameday'].shift(1)).dt.days.fillna(7).clip(upper=14)
    games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'home_rest'}).drop(columns=['team'])
    games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'away_rest'}).drop(columns=['team'])

    cols_base = ['off_epa_long', 'off_ypp_long', 'off_pass_epa_long', 'off_edsr_long', 'off_sack_rate_long', 
                 'def_sack_rate_long', 'st_epa_long', 'turnovers_lost_long', 'pen_yards_long', 'off_rz_epa_long', 
                 'qb_volatility', 'pythag_wins']
    
    for side in ['home', 'away']:
        games = games.merge(stats[['season', 'week', 'team'] + cols_base], left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
        games.rename(columns={c: f'{side}_{c}' for c in cols_base}, inplace=True)
        games.drop(columns=['team'], inplace=True)

    feature_cols = [c for c in games.columns if c not in ['result', 'home_score', 'away_score', 'spread_line', 'game_id', 'season', 'week', 'home_team', 'away_team', 'gameday']]
    games[feature_cols] = games[feature_cols].fillna(0)
    
    games['qb_diff'] = games['home_off_pass_epa_long'] - games['away_off_pass_epa_long']
    games['edsr_diff'] = games['home_off_edsr_long'] - games['away_off_edsr_long']
    games['ypp_diff'] = games['home_off_ypp_long'] - games['away_off_ypp_long']
    games['pythag_diff'] = games['home_pythag_wins'] - games['away_pythag_wins']
    games['rest_diff'] = games['home_rest'] - games['away_rest']
    games['st_diff'] = games['home_st_epa_long'] - games['away_st_epa_long']
    games['turnover_diff'] = games['home_turnovers_lost_long'] - games['away_turnovers_lost_long']
    games['rz_diff'] = games['home_off_rz_epa_long'] - games['away_off_rz_epa_long']
    games['penalty_diff'] = games['home_pen_yards_long'] - games['away_pen_yards_long']
    games['sack_mismatch_home'] = games['home_off_sack_rate_long'] - games['away_def_sack_rate_long']
    games['sack_mismatch_away'] = games['away_off_sack_rate_long'] - games['home_def_sack_rate_long']
    home_results = games.sort_values(['season', 'week'])[['home_team', 'result']].rename(columns={'home_team': 'team'})
    games['home_field_strength'] = home_results.groupby('team')['result'].transform(lambda x: x.shift(1).expanding().mean()).fillna(2.0)
    games['roof'] = games['roof'].map({'outdoors': 0, 'open': 0, 'closed': 1, 'dome': 1}).fillna(0)
    
    return games

if __name__ == "__main__":
    print(f"📥 Downloading Data...")
    schedule = nfl.import_schedules(YEARS)
    pbp = nfl.import_pbp_data(YEARS)

    print("⚙️  Processing Stats...")
    pbp['posteam'] = pbp['posteam'].replace(TEAM_MAP)
    pbp['defteam'] = pbp['defteam'].replace(TEAM_MAP)
    pbp_clean = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & (pbp['wp'] > 0.05) & (pbp['wp'] < 0.95)].dropna(subset=['epa', 'posteam', 'defteam', 'success', 'yards_gained'])
    stats = get_team_stats(pbp_clean, pbp).sort_values(['team', 'season', 'week'])

    metrics = ['off_epa', 'off_ypp', 'off_pass_epa', 'off_edsr', 'off_sack_rate', 'def_sack_rate', 'st_epa', 'turnovers_lost', 'pen_yards', 'off_rz_epa']
    for col in metrics:
        stats[f'{col}_long'] = stats.groupby(['team', 'season'])[col].transform(lambda x: x.shift(1).ewm(span=10).mean())

    games_raw = schedule[schedule['game_type'] == 'REG'].copy()
    games_scored = games_raw.dropna(subset=['home_score', 'away_score'])
    home = games_scored[['season', 'week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'pf', 'away_score': 'pa'})
    away = games_scored[['season', 'week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'pf', 'home_score': 'pa'})
    results = pd.concat([home, away]).sort_values(['team', 'season', 'week'])
    results['team'] = results['team'].replace(TEAM_MAP)
    results['cum_pf'] = results.groupby(['team', 'season'])['pf'].transform(lambda x: x.shift(1).cumsum())
    results['cum_pa'] = results.groupby(['team', 'season'])['pa'].transform(lambda x: x.shift(1).cumsum())
    results['pythag_wins'] = np.where(results['cum_pf'] == 0, 0, (results['cum_pf']**2.37) / ((results['cum_pf']**2.37) + (results['cum_pa']**2.37)))
    stats = stats.merge(results[['season', 'week', 'team', 'pythag_wins']], on=['season', 'week', 'team'], how='left').fillna(0)

    qb_data = pbp_clean[pbp_clean['pass'] == 1].groupby(['season', 'posteam', 'name']).agg({'epa': 'mean', 'play_id': 'count'}).reset_index()
    qb_db = qb_data[qb_data['play_id'] > 15].copy()
    qb_stability = qb_db.groupby(['season', 'posteam'])['epa'].std().reset_index().rename(columns={'epa': 'qb_volatility', 'posteam': 'team'}).fillna(0)
    stats = stats.merge(qb_stability, on=['season', 'team'], how='left')
    full_games_df = engineer_features(schedule, stats, qb_db)

    # --- INTELLIGENT TRAINING CHECK ---
    model = None
    if os.path.exists(MODEL_PATH):
        print("✅ Found existing model (Decrypted). SKIPPING training.")
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
    else:
        print("⚡ No model found. Training fresh model...")
        train_data = full_games_df[full_games_df['season'] < CURRENT_SEASON].dropna(subset=['result'])
        X_cols = [
            'qb_diff', 'edsr_diff', 'ypp_diff', 'pythag_diff', 'rest_diff',
            'sack_mismatch_home', 'sack_mismatch_away',
            'st_diff', 'turnover_diff', 'rz_diff', 'penalty_diff',
            'home_field_strength', 'roof',
            'home_qb_volatility', 'away_qb_volatility'
        ]
        y_train = train_data['result'].clip(-21, 21)
        mono_constraints = (1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 0, -1, 1)
        model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=3, min_child_weight=20, 
                                 reg_alpha=0.5, subsample=0.5, colsample_bytree=0.5, 
                                 monotone_constraints=mono_constraints, n_jobs=-1, objective='reg:squarederror')
        model.fit(train_data[X_cols], y_train)
        model.save_model(MODEL_PATH)

    print(f"📦 Updating Cache: {CACHE_PATH}...")
    db = {
        'model': model,
        'games_df': full_games_df, 
        'current_season': CURRENT_SEASON,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(db, f)

    print("✅ Stats Update Complete.")
'''

def write_file(path, content):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created {path}")

if __name__ == "__main__":
    print("🚀 Optimizing Deployment...")
    write_file("requirements.txt", REQUIREMENTS_TXT)
    write_file(".github/workflows/update_picks.yml", WORKFLOW_CONTENT)
    write_file("update_db.py", UPDATE_DB_CONTENT)
    print("\n✅ Setup Complete.")