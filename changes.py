import os

# ==========================================
# 1. FILE CONTENTS
# ==========================================

REQUIREMENTS_TXT = """pandas
numpy
xgboost
nfl_data_py
scikit-learn
jinja2
tabulate
"""

# --- GITIGNORE: PROTECTS YOUR MODEL ---
GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
venv/
.env

# Model & Data (Keep these private)
nfl_bettor.json
data/nfl_db.pkl
"""

# --- README: PROFESSIONAL DOCUMENTATION ---
README_CONTENT = """# 🏈 NFL XGBoost Handicapper

**Automated Institutional-Grade NFL Betting Model**

[![Model Update](https://github.com/Ducky705/nfl-xgboost-model/actions/workflows/update_picks.yml/badge.svg)](https://github.com/Ducky705/nfl-xgboost-model/actions/workflows/update_picks.yml)
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-brightgreen)](https://ducky705.github.io/nfl-xgboost-model/)

## 📊 Overview
This project is an automated machine learning pipeline that predicts NFL game outcomes against the spread. It utilizes a **XGBoost Regressor** trained on play-by-play data (2018-Present) to identify inefficiencies in Vegas lines.

The system runs completely autonomously via **GitHub Actions**.

## 🧠 Methodology
The model focuses on "Trench Warfare" and "Efficiency" metrics rather than simple points scored.

* **Strict Walk-Forward Validation:** The model is trained *only* on past seasons to prevent data leakage.
* **Key Features:**
    * *EPA/Play Differential (QB & Team)*
    * *Early Down Success Rate (EDSR)*
    * *Sack Rate Mismatches*
    * *Pythagorean Win Expectancy*
* **Unit Sizing:**
    * **🔥 STRONG (2u):** Edge > 3.5 points.
    * **⚠️ LEAN (1u):** Edge > 1.5 points.

## 🚀 How It Works
1.  **Fetch:** Downloads latest NFL play-by-play data.
2.  **Process:** Engineers 15+ advanced efficiency metrics.
3.  **Train:** Retrains the model on strictly historical data (preventing lookahead bias).
4.  **Predict:** Simulates upcoming games and assigns a "Fair Price."
5.  **Publish:** Updates the [Web Dashboard](https://ducky705.github.io/nfl-xgboost-model/) with active picks.

## ⚠️ Disclaimer
This software is for educational and informational purposes only. It does not constitute financial advice. Sports betting involves significant risk.
"""

# --- HEAVY LIFTER: update_db.py ---
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

print(f"🔄 STARTING DB UPDATE for {CURRENT_SEASON}...")

def get_team_stats(df, full_pbp):
    print("   -> Calculating Stats...")
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
    print("   -> Merging & Engineering Features...")
    games = schedule[schedule['game_type'] == 'REG'].copy()
    
    # Fix Names
    games['home_team'] = games['home_team'].replace(TEAM_MAP)
    games['away_team'] = games['away_team'].replace(TEAM_MAP)
    
    games = games.drop(columns=['home_rest', 'away_rest'], errors='ignore')

    # Rest
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

    # --- CRITICAL FIX: DO NOT FILLNA 'result' ---
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

# --- EXECUTION ---
if __name__ == "__main__":
    print(f"📥 Downloading Data...")
    try:
        schedule = nfl.import_schedules(YEARS)
        pbp = nfl.import_pbp_data(YEARS)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    print("⚙️  Processing...")
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

    print("🧠 Building Master DB...")
    full_games_df = engineer_features(schedule, stats, qb_db)

    print(f"⚡ Training Model on Data PRIOR to {CURRENT_SEASON}...")
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
    print("💾 Honest Model Saved.")

    print(f"📦 Saving to {CACHE_PATH}...")
    db = {
        'model': model,
        'games_df': full_games_df, 
        'current_season': CURRENT_SEASON,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    }
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(db, f)

    print("✅ DONE. Run 'python main.py' to generate dashboard.")
'''

# --- DASHBOARD: main.py ---
MAIN_PY_CONTENT = r'''import pandas as pd
import pickle
import os
import webbrowser
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate
from datetime import datetime

# --- CONFIG ---
CACHE_PATH = "data/nfl_db.pkl"
DATA_PATH = "data/betting_history.csv"
DOCS_PATH = "docs/index.html"
FIX_VEGAS_SIGNS = True 

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL AI Handicapper</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { background-color: #1e293b; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .win { color: #4ade80; }
        .loss { color: #f87171; }
        .push { color: #94a3b8; }
        .table-row:nth-child(even) { background-color: #334155; }
        .strong-bet { border-left: 4px solid #ef4444; background-color: #2c364c; }
    </style>
</head>
<body class="p-6 max-w-7xl mx-auto">
    
    <div class="mb-8 text-center">
        <h1 class="text-4xl font-bold text-blue-400 mb-2">🏈 NFL AI Handicapper</h1>
        <p class="text-gray-400">Automated predictions using XGBoost & Walk-Forward Validation</p>
        <p class="text-xs text-gray-500 mt-2">Last Updated: {{ last_updated }}</p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div class="card text-center">
            <h3 class="text-gray-400 text-sm">Total ROI</h3>
            <p class="text-3xl font-bold {{ 'win' if roi > 0 else 'loss' }}">{{ roi }}%</p>
        </div>
        <div class="card text-center">
            <h3 class="text-gray-400 text-sm">Units Won</h3>
            <p class="text-3xl font-bold {{ 'win' if units > 0 else 'loss' }}">{{ units }}u</p>
        </div>
        <div class="card text-center">
            <h3 class="text-gray-400 text-sm">Record</h3>
            <p class="text-3xl font-bold">{{ record }}</p>
        </div>
        <div class="card text-center">
            <h3 class="text-gray-400 text-sm">Win Rate</h3>
            <p class="text-3xl font-bold">{{ win_pct }}%</p>
        </div>
    </div>

    <h2 class="text-2xl font-bold mb-4 border-b border-gray-700 pb-2">📋 This Week's Card</h2>
    <div class="overflow-x-auto rounded-lg shadow-lg mb-12">
        <table class="w-full text-left text-sm text-gray-300">
            <thead class="bg-gray-800 text-xs uppercase font-medium">
                <tr>
                    <th class="px-4 py-3">Matchup</th>
                    <th class="px-4 py-3">Spread</th>
                    <th class="px-4 py-3">Model Line</th>
                    <th class="px-4 py-3">Edge</th>
                    <th class="px-4 py-3">Recommendation</th>
                    <th class="px-4 py-3">Confidence</th>
                </tr>
            </thead>
            <tbody class="bg-gray-900">
                {% for bet in active_bets %}
                <tr class="table-row border-b border-gray-800 {{ 'strong-bet' if 'STRONG' in bet.Conf else '' }}">
                    <td class="px-4 py-4 font-medium">{{ bet.Matchup }}</td>
                    <td class="px-4 py-4">{{ bet.Vegas }}</td>
                    <td class="px-4 py-4">{{ bet.Fair_Line }}</td>
                    <td class="px-4 py-4 font-bold win">+{{ bet.Edge }}</td>
                    <td class="px-4 py-4 font-bold text-white">{{ bet.Action }}</td>
                    <td class="px-4 py-4">
                        {% if bet.Conf %}
                        <span class="px-2 py-1 rounded text-xs font-bold 
                        {{ 'bg-red-900 text-red-200' if 'STRONG' in bet.Conf else 'bg-yellow-900 text-yellow-200' }}">
                            {{ bet.Conf }}
                        </span>
                        {% else %}
                        <span class="px-2 py-1 rounded text-xs font-bold bg-gray-700 text-gray-400">None</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% if active_bets|length == 0 %}
        <div class="p-8 text-center text-gray-500 bg-gray-900">No active value bets found for the upcoming week yet.</div>
        {% endif %}
    </div>

    <h2 class="text-2xl font-bold mb-4 border-b border-gray-700 pb-2">📜 Recent Graded Bets</h2>
    <div class="overflow-x-auto rounded-lg shadow-lg">
        <table class="w-full text-left text-sm text-gray-400">
            <thead class="bg-gray-800 text-xs uppercase">
                <tr>
                    <th class="px-4 py-3">Week</th>
                    <th class="px-4 py-3">Matchup</th>
                    <th class="px-4 py-3">Spread</th>
                    <th class="px-4 py-3">Model</th>
                    <th class="px-4 py-3">Edge</th>
                    <th class="px-4 py-3">Pick</th>
                    <th class="px-4 py-3">Conf</th>
                    <th class="px-4 py-3">Result</th>
                    <th class="px-4 py-3">Profit</th>
                </tr>
            </thead>
            <tbody>
                {% for bet in history %}
                <tr class="table-row border-b border-gray-800">
                    <td class="px-4 py-3">W{{ bet.week }}</td>
                    <td class="px-4 py-3">{{ bet.matchup }}</td>
                    <td class="px-4 py-3">{{ bet.line_display }}</td>
                    <td class="px-4 py-3">{{ bet.fair_display }}</td>
                    <td class="px-4 py-3 win">+{{ bet.edge }}</td>
                    <td class="px-4 py-3 font-bold">{{ bet.pick }}</td>
                    <td class="px-4 py-3">
                        {% if bet.conf %}
                        <span class="px-1 py-0.5 rounded text-xs font-bold 
                        {{ 'bg-red-900 text-red-200' if 'STRONG' in bet.conf else 'bg-yellow-900 text-yellow-200' }}">
                            {{ bet.conf }}
                        </span>
                        {% else %}
                        <span class="px-2 py-1 rounded text-xs font-bold bg-gray-700 text-gray-400">-</span>
                        {% endif %}
                    </td>
                    <td class="px-4 py-3">
                        <span class="{{ 'win' if bet.result == 'WIN' else ('loss' if bet.result == 'LOSS' else 'push') }}">
                            {{ bet.result }}
                        </span>
                    </td>
                    <td class="px-4 py-3 {{ 'win' if bet.profit > 0 else ('loss' if bet.profit < 0 else 'push') }}">{{ bet.profit }}u</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

# --- 1. LOAD DB ---
if not os.path.exists(CACHE_PATH):
    print("❌ Cache not found!")
    exit()

with open(CACHE_PATH, 'rb') as f:
    db = pickle.load(f)

model = db['model']
games_df = db['games_df']
CURRENT_SEASON = db['current_season']

X_cols = [
    'qb_diff', 'edsr_diff', 'ypp_diff', 'pythag_diff', 'rest_diff',
    'sack_mismatch_home', 'sack_mismatch_away',
    'st_diff', 'turnover_diff', 'rz_diff', 'penalty_diff',
    'home_field_strength', 'roof',
    'home_qb_volatility', 'away_qb_volatility'
]

# --- 2. BACKFILL ENGINE ---
def run_backfill(model, games_df):
    completed = games_df[(games_df['season'] == CURRENT_SEASON) & (games_df['result'].notna())].copy()
    
    bets = []
    for _, game in completed.iterrows():
        X = pd.DataFrame([game[X_cols]])
        raw_pred = -1 * model.predict(X)[0]
        fair_line = max(min(raw_pred, 21), -21)
        
        raw_spread = game['spread_line']
        if pd.isna(raw_spread): continue
        vegas_line = -1 * raw_spread if FIX_VEGAS_SIGNS else raw_spread
        
        diff = fair_line - vegas_line
        if abs(diff) > 10: fair_line = vegas_line + (diff * 0.5) 
        
        # --- POSITIVE EDGE LOGIC ---
        raw_edge = round(vegas_line - fair_line, 2)
        abs_edge = abs(raw_edge)
        
        action = "PASS"; conf = ""; units = 0.0
        
        if abs_edge >= 3.5:
            action = f"BET {game['home_team']}" if raw_edge > 0 else f"BET {game['away_team']}"
            conf = "STRONG"
            units = 2.0
        elif abs_edge >= 1.5:
            action = f"BET {game['home_team']}" if raw_edge > 0 else f"BET {game['away_team']}"
            conf = "LEAN"
            units = 1.0
            
        if "BET" in action:
            pick = action.replace("BET ", "")
            actual = game['result']
            
            res = 'LOSS'; profit = -1.0 * units
            
            if pick == game['home_team']:
                if actual > game['spread_line']: 
                    res = 'WIN'; profit = 0.91 * units
                elif actual == game['spread_line']: 
                    res = 'PUSH'; profit = 0.0
            else:
                if actual < game['spread_line']: 
                    res = 'WIN'; profit = 0.91 * units
                elif actual == game['spread_line']: 
                    res = 'PUSH'; profit = 0.0
            
            vegas_dsp = f"{game['home_team']} {vegas_line:.1f}" if vegas_line < 0 else f"{game['home_team']} +{vegas_line:.1f}"
            fair_dsp = f"{game['home_team']} {fair_line:.1f}" if fair_line < 0 else f"{game['home_team']} +{fair_line:.1f}"

            bets.append({
                'season': game['season'], 'week': game['week'], 
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'pick': f"{pick} ({int(units)}u)", 
                'line_display': vegas_dsp, 'fair_display': fair_dsp,
                'edge': round(abs_edge, 1),
                'conf': conf,
                'result': res, 'profit': round(profit, 2),
                'units_wagered': units,
                'status': 'GRADED', 'game_id': game['game_id']
            })
            
    return pd.DataFrame(bets)

# --- 3. PREDICT UPCOMING ---
def run_predictions(model, games_df):
    upcoming = games_df[(games_df['season'] == CURRENT_SEASON) & (games_df['result'].isna())].copy()
    if upcoming.empty: return []
    
    next_week = upcoming['week'].min()
    print(f"🔮 Analyzing Week {next_week}...")
    week_df = upcoming[upcoming['week'] == next_week]
    
    preds = []
    for _, game in week_df.iterrows():
        X = pd.DataFrame([game[X_cols]])
        raw_pred = -1 * model.predict(X)[0]
        fair_line = max(min(raw_pred, 21), -21)
        
        raw_spread = game['spread_line']
        if pd.isna(raw_spread): continue
        vegas_line = -1 * raw_spread if FIX_VEGAS_SIGNS else raw_spread
        
        diff = fair_line - vegas_line
        if abs(diff) > 10: fair_line = vegas_line + (diff * 0.5)
        
        # --- POSITIVE EDGE LOGIC ---
        raw_edge = round(vegas_line - fair_line, 2)
        abs_edge = abs(raw_edge)
        
        action = "PASS"; conf = ""; units = 0
        if abs_edge >= 3.5: 
            units = 2
            action = f"BET {game['home_team']} ({units}u)" if raw_edge > 0 else f"BET {game['away_team']} ({units}u)"
            conf = "🔥 STRONG"
        elif abs_edge >= 1.5: 
            units = 1
            action = f"BET {game['home_team']} ({units}u)" if raw_edge > 0 else f"BET {game['away_team']} ({units}u)"
            conf = "⚠️ LEAN"
        
        vegas_dsp = f"{game['home_team']} {vegas_line:.1f}" if vegas_line < 0 else f"{game['home_team']} +{vegas_line:.1f}"
        fair_dsp = f"{game['home_team']} {fair_line:.1f}" if fair_line < 0 else f"{game['home_team']} +{fair_line:.1f}"
        
        preds.append({
            'Matchup': f"{game['away_team']} @ {game['home_team']}",
            'Vegas': vegas_dsp, 'Fair_Line': fair_dsp, 
            'Edge': round(abs_edge, 1),
            'Action': action, 'Conf': conf, 'week': next_week, 'game_id': game['game_id'],
            'units': units,
            'pick': action.replace("BET ", "")
        })
    return preds

# --- 4. OUTPUT ---
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    backfill_df = run_backfill(model, games_df)
    active_bets = run_predictions(model, games_df)
    
    # Sort Active Bets: Units Desc, then Edge Desc
    if active_bets:
        active_bets.sort(key=lambda x: (x['units'], x['Edge']), reverse=True)
        print("\n" + tabulate(pd.DataFrame(active_bets)[['Matchup', 'Vegas', 'Fair_Line', 'Edge', 'Action']], headers="keys", tablefmt="github"))
    
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH)
        existing = existing[existing['season'] != CURRENT_SEASON]
    else:
        existing = pd.DataFrame()
        
    final_hist = pd.concat([existing, backfill_df], ignore_index=True)
    final_hist.to_csv(DATA_PATH, index=False)
    
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('dashboard.html')
    
    curr = final_hist[final_hist['season'] == CURRENT_SEASON]
    graded = curr[curr['status'] == 'GRADED']
    wins = len(graded[graded['result'] == 'WIN'])
    losses = len(graded[graded['result'] == 'LOSS'])
    pushes = len(graded[graded['result'] == 'PUSH'])
    total = wins + losses + pushes
    profit = graded['profit'].sum()
    
    recent = []
    if not graded.empty:
        last_wk = graded['week'].max()
        # Sort Recent Bets: Profit Desc
        recent_df = graded[graded['week'] == last_wk].copy()
        recent_df = recent_df.sort_values('profit', ascending=False)
        recent = recent_df.to_dict('records')
        
    html = template.render(
        active_bets=active_bets,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        roi=round((profit/total)*100, 1) if total > 0 else 0.0,
        units=round(profit, 2),
        record=f"{wins}-{losses}-{pushes}",
        win_pct=round((wins/total)*100, 1) if total > 0 else 0.0,
        history=recent
    )
    
    with open(DOCS_PATH, "w", encoding="utf-8") as f: f.write(html)
    full_path = os.path.abspath(DOCS_PATH)
    print(f"✅ Dashboard Updated: {full_path}")
    webbrowser.open(f"file://{full_path}")
'''

# --- GITHUB WORKFLOW: AUTOMATIC UPDATES ---
WORKFLOW_CONTENT = """name: NFL Bot Auto-Update

on:
  schedule:
    # Run heavy data update (training) only on Tuesdays at 8:00 AM UTC
    - cron: '0 8 * * 2'
    # Run dashboard update (odds & grading) Daily at 9:00 AM UTC
    - cron: '0 9 * * *'
  workflow_dispatch:      # Allows manual "Click to Run" button

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

    - name: Run Heavy Update (Always run to generate model ephemerally)
      run: python update_db.py

    - name: Run Dashboard Generation
      run: python main.py

    - name: Commit and Push Changes
      run: |
        git config --global user.name "NFL-Bot"
        git config --global user.email "bot@nfl.com"
        git add data/ docs/
        git commit -m "🏈 Daily Bot Update" || echo "No changes to commit"
        git push
"""

# ==========================================
# 2. WRITE FILES
# ==========================================

def write_file(path, content):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created {path}")

if __name__ == "__main__":
    print("🚀 Initializing Project Setup...")
    
    # 1. Requirements
    write_file("requirements.txt", REQUIREMENTS_TXT)
    
    # 2. Scripts
    write_file("update_db.py", UPDATE_DB_CONTENT)
    write_file("main.py", MAIN_PY_CONTENT)
    
    # 3. Automation
    write_file(".github/workflows/update_picks.yml", WORKFLOW_CONTENT)
    
    # 4. Gitignore (Critical)
    write_file(".gitignore", GITIGNORE_CONTENT)
    
    # 5. Readme
    write_file("README.md", README_CONTENT)
    
    print("\n🎉 Project Setup Complete!")
    print("-------------------------------------------------------")
    print("1. git init")
    print("2. git add .")
    print("3. git commit -m 'Initial commit'")
    print("4. git branch -M main")
    print("5. git remote add origin https://github.com/Ducky705/nfl-xgboost-model.git")
    print("6. git push -u origin main")
    print("-------------------------------------------------------")