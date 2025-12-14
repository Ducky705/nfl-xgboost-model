import pandas as pd
import pickle
import os
import webbrowser
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate
from datetime import datetime
from scipy.stats import norm

# --- CONFIG ---
CACHE_PATH = "data/nfl_db.pkl"
DATA_PATH = "data/betting_history.csv"
DOCS_PATH = "docs/index.html"
FIX_VEGAS_SIGNS = True 

# --- KELLY CRITERION FUNCTION ---
def calculate_kelly_units(abs_edge):
    """Calculates Kelly Unit size based on point spread edge."""
    # Constants
    STD_DEV = 13.86           # Standard deviation of NFL spread errors
    PAYOUT_RATIO = 0.9091     # Net odds for -110 (10/11)
    MIN_EDGE = 1.0            # Minimum edge to bet
    MAX_UNITS = 2.0           # Strict Max Bet
    
    # 0.05 (5% Kelly) ensures only massive edges hit the 2.0u cap
    KELLY_FRACTION = 0.05     

    if abs_edge < MIN_EDGE:
        return 0.0, "None"

    # 1. Win Probability (p)
    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    # 2. Kelly Fraction (K)
    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    
    # Apply 5% Kelly
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    
    # 3. Rounding & Capping
    units = round(min(kelly_units, MAX_UNITS), 1)
    
    # Dynamic Confidence Labels based on Unit Size
    if units >= 1.5: 
        conf = "🔥 STRONG"
    elif units >= 0.8:
        conf = "💪 SOLID" 
    elif units >= 0.1:
        conf = "⚠️ LEAN"
    else:
        conf = "None"
        
    return units, conf


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
                        {% if 'None' not in bet.Conf %}
                        <span class="px-2 py-1 rounded text-xs font-bold 
                        {{ 'bg-red-900 text-red-200' if 'STRONG' in bet.Conf else ('bg-green-900 text-green-200' if 'SOLID' in bet.Conf else 'bg-yellow-900 text-yellow-200') }}">
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
                        {% if 'None' not in bet.conf %}
                        <span class="px-1 py-0.5 rounded text-xs font-bold 
                        {{ 'bg-red-900 text-red-200' if 'STRONG' in bet.conf else ('bg-green-900 text-green-200' if 'SOLID' in bet.conf else 'bg-yellow-900 text-yellow-200') }}">
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
        
        # --- KELLY CRITERION LOGIC ---
        raw_edge = round(vegas_line - fair_line, 2)
        abs_edge = abs(raw_edge)
        
        units, conf_badge = calculate_kelly_units(abs_edge)
        
        action = "PASS"; conf = ""; 
        
        if units > 0.0:
            pick_team = game['home_team'] if raw_edge > 0 else game['away_team']
            action = f"BET {pick_team}"
            # Remove emojis for history table
            conf = conf_badge.replace("🔥 ", "").replace("⚠️ ", "").replace("💪 ", "").replace("None", "")
            
            pick = action.replace("BET ", "")
            actual = game['result']
            
            res = 'LOSS'; profit = -1.0 * units
            
            if pick == game['home_team']:
                if actual > game['spread_line']: 
                    res = 'WIN'; profit = round(0.91 * units, 2)
                elif actual == game['spread_line']: 
                    res = 'PUSH'; profit = 0.0
            else:
                if actual < game['spread_line']: 
                    res = 'WIN'; profit = round(0.91 * units, 2)
                elif actual == game['spread_line']: 
                    res = 'PUSH'; profit = 0.0
            
            vegas_dsp = f"{game['home_team']} {vegas_line:.1f}" if vegas_line < 0 else f"{game['home_team']} +{vegas_line:.1f}"
            fair_dsp = f"{game['home_team']} {fair_line:.1f}" if fair_line < 0 else f"{game['home_team']} +{fair_line:.1f}"

            bets.append({
                'season': game['season'], 'week': game['week'], 
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'pick': f"{pick} ({units}u)", 
                'line_display': vegas_dsp, 'fair_display': fair_dsp,
                'edge': round(abs_edge, 1),
                'conf': conf,
                'result': res, 'profit': profit,
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
        
        # --- KELLY CRITERION LOGIC ---
        raw_edge = round(vegas_line - fair_line, 2)
        abs_edge = abs(raw_edge)
        
        units, conf_badge = calculate_kelly_units(abs_edge)
        
        action = "PASS"; conf = conf_badge; 
        
        if units > 0.0: 
            pick_team = game['home_team'] if raw_edge > 0 else game['away_team']
            action = f"BET {pick_team} ({units}u)"
        
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
