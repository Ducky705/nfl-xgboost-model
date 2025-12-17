import pandas as pd
import pickle
import os
import webbrowser
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate
from datetime import datetime
from scipy.stats import norm
import xgboost as xgb  # Required for DMatrix

# --- CONFIG ---
CACHE_PATH = "data/nfl_db.pkl"
DATA_PATH = "data/betting_history.csv"
DOCS_PATH = "docs/index.html"
FIX_VEGAS_SIGNS = True 

# --- KELLY CRITERION FUNCTION ---
def calculate_kelly_units(abs_edge):
    """Calculates Kelly Unit size based on point spread edge."""
    STD_DEV = 13.86
    PAYOUT_RATIO = 0.9091
    MIN_EDGE = 1.0
    MAX_UNITS = 2.0
    KELLY_FRACTION = 0.05

    if abs_edge < MIN_EDGE:
        return 0.0, "None"

    z_score = abs_edge / STD_DEV
    p = norm.cdf(z_score) 
    q = 1.0 - p

    full_kelly_percent = (p - (q / PAYOUT_RATIO)) * 100
    kelly_units = max(0.0, full_kelly_percent * KELLY_FRACTION)
    units = round(min(kelly_units, MAX_UNITS), 1)
    
    if units >= 1.5: conf = "🔥 STRONG"
    elif units >= 0.8: conf = "💪 SOLID" 
    elif units >= 0.1: conf = "⚠️ LEAN"
    else: conf = "None"
        
    return units, conf

# --- SYSTEM CONFIDENCE FUNCTION ---
def calculate_system_confidence(graded_df):
    """Calculates system confidence based on win rate variance across weeks."""
    if graded_df.empty or len(graded_df) < 5:
        return "CALIBRATING", "text-zinc-500", "Insufficient data"
    
    # Group by week and calculate win rate per week
    weekly_stats = graded_df.groupby('week').apply(
        lambda x: len(x[x['result'] == 'WIN']) / len(x) * 100 if len(x) > 0 else 0
    )
    
    if len(weekly_stats) < 3:
        return "CALIBRATING", "text-zinc-500", "Need more weeks"
    
    # Calculate variance and standard deviation
    variance = weekly_stats.var()
    std_dev = weekly_stats.std()
    mean_wr = weekly_stats.mean()
    
    # Determine confidence level based on std deviation
    # Lower variance = more stable/confident system
    if std_dev < 10:
        return "STABLE", "text-acid-lime", f"σ={std_dev:.1f}%"
    elif std_dev < 20:
        return "MODERATE", "text-zinc-300", f"σ={std_dev:.1f}%"
    elif std_dev < 30:
        return "ELEVATED", "text-yellow-500", f"σ={std_dev:.1f}%"
    else:
        return "HIGH VOLATILITY", "text-warning-orange", f"σ={std_dev:.1f}%"

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PROTOCOL 705 // ALGORITHMIC MARKETS</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;800&family=Space+Grotesk:wght@300;500;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Custom Config -->
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'void': '#050505',
                        'panel': '#0A0A0A',
                        'border-dim': '#1F1F1F',
                        'acid-lime': '#CCFF00', 
                        'warning-orange': '#FF4D00', 
                        'ghost': '#444444'
                    },
                    fontFamily: {
                        'sans': ['"Space Grotesk"', 'sans-serif'],
                        'mono': ['"JetBrains Mono"', 'monospace'],
                    },
                    animation: {
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'ticker': 'ticker 30s linear infinite',
                    },
                    keyframes: {
                        ticker: {
                            '0%': { transform: 'translateX(0)' },
                            '100%': { transform: 'translateX(-100%)' },
                        }
                    }
                }
            }
        }
    </script>

    <style>
        body {
            background-color: #050505;
            color: #E2E2E2;
            -webkit-font-smoothing: antialiased;
        }

        .noise {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E");
            pointer-events: none;
            z-index: 50;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0A0A0A; }
        ::-webkit-scrollbar-thumb { background: #333; }

        .glow-text { text-shadow: 0 0 10px rgba(204, 255, 0, 0.3); }
        
        .bg-grid {
            background-size: 40px 40px;
            background-image: linear-gradient(to right, #1a1a1a 1px, transparent 1px),
                              linear-gradient(to bottom, #1a1a1a 1px, transparent 1px);
        }

        .signal-row {
            cursor: pointer;
        }
        .signal-row:hover {
            transform: translateX(4px);
            box-shadow: 0 0 20px rgba(204, 255, 0, 0.15);
        }
    </style>
</head>
<body class="min-h-screen relative overflow-x-hidden">
    <div class="noise"></div>

    <!-- 1. THE TICKER TAPE -->
    <div class="fixed top-0 w-full bg-acid-lime text-black font-mono text-xs font-bold py-1 z-40 overflow-hidden whitespace-nowrap border-b border-acid-lime">
        <div class="inline-block animate-ticker">
            PROTOCOL 705 ONLINE // LAST UPDATE: {{ last_updated }} // ROI: {{ roi }}% // RECORD: {{ record }} // ALPHA GENERATED: {{ units }}u // SYSTEM LOAD: NOMINAL // MARKET VOLATILITY: DETECTED // 
            PROTOCOL 705 ONLINE // LAST UPDATE: {{ last_updated }} // ROI: {{ roi }}% // RECORD: {{ record }} // ALPHA GENERATED: {{ units }}u // SYSTEM LOAD: NOMINAL // MARKET VOLATILITY: DETECTED // 
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <div class="max-w-7xl mx-auto pt-16 pb-20 px-4 sm:px-6">

        <!-- 2. HEADER SECTION -->
        <header class="mb-12 flex flex-col md:flex-row justify-between items-end border-b border-border-dim pb-6">
            <div>
                <div class="flex items-center gap-2 mb-2">
                    <div class="w-3 h-3 bg-acid-lime rounded-full animate-pulse-slow"></div>
                    <span class="font-mono text-xs text-acid-lime tracking-widest">LIVE SIGNAL</span>
                </div>
                <h1 class="text-5xl md:text-7xl font-bold tracking-tighter text-white">PROTOCOL <span class="text-ghost">705</span></h1>
                <p class="font-mono text-zinc-500 mt-2 text-sm uppercase tracking-wide">XGBoost Algorithmic Infrastructure</p>
            </div>
            <div class="text-right mt-6 md:mt-0">
                <div class="font-mono text-4xl font-bold text-acid-lime glow-text">{{ "+" if units > 0 else "" }}{{ units }}u</div>
                <div class="font-mono text-xs text-zinc-500 uppercase tracking-widest">Net Alpha Generated</div>
            </div>
        </header>

        <!-- 3. THE BENTO GRID DASHBOARD -->
        <section class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-16">
            <!-- Metric 1: ROI -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-colors duration-300 group">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase">Return on Investment</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                </div>
                <div class="text-3xl font-bold text-white group-hover:text-acid-lime transition-colors">{{ roi }}%</div>
            </div>

            <!-- Metric 2: Win Rate -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-colors duration-300 group">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase">Win Rate</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                </div>
                <div class="flex items-end gap-3">
                    <div class="text-3xl font-bold text-white group-hover:text-acid-lime transition-colors">{{ win_pct }}%</div>
                    <div class="h-1 flex-1 bg-zinc-800 mb-2 rounded-full overflow-hidden">
                        <div class="h-full bg-white group-hover:bg-acid-lime transition-colors" style="width: {{ win_pct }}%"></div>
                    </div>
                </div>
            </div>

            <!-- Metric 3: Record -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 hover:border-acid-lime transition-colors duration-300 group">
                <div class="flex justify-between items-start">
                    <span class="font-mono text-xs text-zinc-500 uppercase">W - L - T</span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
                <div class="text-3xl font-mono text-zinc-300 group-hover:text-acid-lime transition-colors">{{ record | replace('-', '<span class="text-zinc-600 mx-1">/</span>') | safe }}</div>
            </div>

            <!-- Metric 4: System Confidence -->
            <div class="bg-panel border border-border-dim p-6 flex flex-col justify-between h-32 relative overflow-hidden hover:border-acid-lime transition-colors duration-300 group">
                <div class="flex justify-between items-start z-10 relative">
                    <span class="font-mono text-xs text-zinc-500 uppercase">System Confidence <span class="text-zinc-600">({{ confidence_detail }})</span></span>
                    <svg class="w-4 h-4 text-acid-lime opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                </div>
                <div class="text-xl font-bold {{ confidence_color }} group-hover:text-acid-lime z-10 relative transition-colors">{{ confidence_level }}</div>
                <!-- Visual Decoration: Wave -->
                <svg class="absolute bottom-0 left-0 w-full h-20 text-zinc-900 z-0" fill="currentColor" viewBox="0 0 1440 320" preserveAspectRatio="none"><path fill-opacity="1" d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,224C672,245,768,267,864,250.7C960,235,1056,181,1152,165.3C1248,149,1344,171,1392,181.3L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path></svg>
            </div>
        </section>

        <!-- 4. THE TRADING DESK (Picks) -->
        <main class="mb-20">
            <div class="flex items-center justify-between mb-6">
                <h2 class="font-sans text-2xl font-bold text-white tracking-tight">ACTIVE SIGNALS</h2>
                <div class="flex gap-2" id="filter-buttons">
                    <button data-filter="all" class="filter-btn active px-2 py-1 border border-acid-lime text-acid-lime font-mono text-[10px] uppercase cursor-pointer hover:bg-acid-lime hover:text-black transition-colors">All</button>
                    <button data-filter="strong" class="filter-btn px-2 py-1 border border-zinc-700 text-zinc-500 font-mono text-[10px] uppercase cursor-pointer hover:border-acid-lime hover:text-acid-lime transition-colors">Strong Buy</button>
                    <button data-filter="solid" class="filter-btn px-2 py-1 border border-zinc-700 text-zinc-500 font-mono text-[10px] uppercase cursor-pointer hover:border-acid-lime hover:text-acid-lime transition-colors">Solid</button>
                    <button data-filter="lean" class="filter-btn px-2 py-1 border border-zinc-700 text-zinc-500 font-mono text-[10px] uppercase cursor-pointer hover:border-acid-lime hover:text-acid-lime transition-colors">Lean</button>
                </div>
            </div>

            <!-- Table Header (Styled as Grid) -->
            <div class="hidden md:grid grid-cols-12 gap-4 text-xs font-mono text-zinc-500 uppercase tracking-wider mb-4 px-6">
                <div class="col-span-3">Instrument (Matchup)</div>
                <div class="col-span-2 text-right">Market Line</div>
                <div class="col-span-2 text-right">Model Val</div>
                <div class="col-span-3">Alpha Gap (Edge)</div>
                <div class="col-span-2 text-right">Execution</div>
            </div>

            <!-- Dynamic Content Container -->
            <div id="cards-container" class="space-y-1">
                {% for game in active_bets %}
                {% set borderClass = "border-border-dim" %}
                {% set barColor = "bg-zinc-600" %}
                {% set btnClass = "bg-zinc-800 text-zinc-400" %}
                {% set rowOpacity = "opacity-80 hover:opacity-100" %}
                {% set confidence = "lean" %}

                {% if 'STRONG' in game.clean_conf %}
                    {% set borderClass = "border-l-4 border-l-acid-lime border-y border-r border-border-dim bg-[#0F110A]" %}
                    {% set barColor = "bg-acid-lime" %}
                    {% set btnClass = "bg-acid-lime text-black hover:bg-white transition-colors" %}
                    {% set rowOpacity = "opacity-100" %}
                    {% set confidence = "strong" %}
                {% elif 'SOLID' in game.clean_conf %}
                    {% set borderClass = "border-l-4 border-l-emerald-700 border-y border-r border-border-dim" %}
                    {% set barColor = "bg-emerald-500" %}
                    {% set btnClass = "bg-emerald-900 text-emerald-100 border border-emerald-700" %}
                    {% set confidence = "solid" %}
                {% endif %}
                
                {% set edgePercent = ((game.Edge / 12) * 100)|round|int %}
                {% if edgePercent > 100 %}{% set edgePercent = 100 %}{% endif %}

                <div data-confidence="{{ confidence }}" class="signal-row grid grid-cols-1 md:grid-cols-12 gap-4 p-4 md:p-6 {{ borderClass }} items-center {{ rowOpacity }} transition-all duration-300">
                    
                    <!-- Matchup -->
                    <div class="col-span-3">
                        <div class="flex items-baseline gap-2">
                            <span class="text-xl md:text-2xl font-bold text-white tracking-tight">{{ game.away }}</span>
                            <span class="text-zinc-600 text-sm">@</span>
                            <span class="text-xl md:text-2xl font-bold text-white tracking-tight">{{ game.home }}</span>
                        </div>
                    </div>

                    <!-- Market Line -->
                    <div class="col-span-2 md:text-right font-mono text-zinc-400">
                        {{ game.Vegas }}
                    </div>

                    <!-- Model Line -->
                    <div class="col-span-2 md:text-right font-mono text-zinc-300">
                        {{ game.Fair_Line }}
                    </div>

                    <!-- Edge Visualization -->
                    <div class="col-span-3 flex flex-col justify-center h-full">
                        <div class="flex justify-between text-xs font-mono mb-1">
                            <span class="text-zinc-500">EDGE</span>
                            <span class="{{ 'text-acid-lime' if 'STRONG' in game.clean_conf else 'text-zinc-300' }} font-bold">+{{ game.Edge }}</span>
                        </div>
                        <div class="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                            <div class="h-full {{ barColor }} shadow-[0_0_10px_rgba(204,255,0,0.5)]" style="width: {{ edgePercent }}%"></div>
                        </div>
                    </div>

                    <!-- Action Button -->
                    <div class="col-span-2 md:text-right mt-2 md:mt-0">
                        <button class="w-full md:w-auto px-4 py-2 font-mono text-xs font-bold uppercase tracking-wider {{ btnClass }}">
                            BET {{ game.pick }} <span class="opacity-70 ml-1">({{ game.units }}u)</span>
                        </button>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            {% if active_bets|length == 0 %}
            <div class="p-8 text-center text-zinc-500 border border-border-dim bg-panel font-mono text-sm">NO ACTIVE SIGNALS DETECTED. SYSTEM STANDBY.</div>
            {% endif %}
        </main>

        <!-- 5. HISTORICAL LEDGER -->
        <section class="border-t border-border-dim pt-12">
            <h2 class="font-sans text-xl font-bold text-zinc-400 mb-6 tracking-tight">GRADED EXECUTION LOG</h2>
            
            <div class="overflow-x-auto">
                <table class="w-full font-mono text-sm text-left">
                    <thead class="text-xs text-zinc-600 uppercase border-b border-border-dim">
                        <tr>
                            <th class="py-3 pl-4">Week</th>
                            <th class="py-3">Matchup</th>
                            <th class="py-3 text-right">Spread</th>
                            <th class="py-3 text-right">Edge</th>
                            <th class="py-3 pl-8">Position</th>
                            <th class="py-3 text-right">Result</th>
                            <th class="py-3 text-right pr-4">P&L</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-border-dim text-zinc-400" id="ledger-body">
                        {% for row in history %}
                        {% set isWin = row.result == 'WIN' %}
                        {% set profitClass = "text-acid-lime" if isWin else ("text-warning-orange" if row.result == 'LOSS' else "text-zinc-500") %}
                        <tr class="hover:bg-zinc-900 transition-colors">
                            <td class="py-3 pl-4 border-b border-zinc-900">W{{ row.week }}</td>
                            <td class="py-3 border-b border-zinc-900 font-bold text-zinc-300">{{ row.matchup }}</td>
                            <td class="py-3 text-right border-b border-zinc-900">{{ row.line_display }}</td>
                            <td class="py-3 text-right border-b border-zinc-900 text-zinc-500">{{ row.edge }}</td>
                            <td class="py-3 pl-8 border-b border-zinc-900"><span class="bg-zinc-800 text-white px-2 py-0.5 text-xs font-bold">{{ row.pick_team }}</span></td>
                            <td class="py-3 text-right border-b border-zinc-900">
                                <span class="{{ profitClass }} font-bold">{{ row.result }}</span>
                            </td>
                            <td class="py-3 text-right pr-4 border-b border-zinc-900 {{ profitClass }}">{{ "+" if row.profit > 0 else "" }}{{ row.profit }}u</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- FOOTER -->
        <footer class="mt-20 border-t border-border-dim pt-8 text-center md:text-left flex flex-col md:flex-row justify-between items-center text-zinc-600 text-xs font-mono">
            <p>PROTOCOL 705 © 2025. DESIGNED BY THE VOID.</p>
            <p class="mt-2 md:mt-0">PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE ALPHA.</p>
        </footer>
    </div>

    <script>
        // Filter functionality
        document.addEventListener('DOMContentLoaded', function() {
            const filterButtons = document.querySelectorAll('.filter-btn');
            const signalRows = document.querySelectorAll('.signal-row');

            filterButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const filter = this.dataset.filter;

                    // Update active button styles
                    filterButtons.forEach(btn => {
                        btn.classList.remove('active', 'border-acid-lime', 'text-acid-lime');
                        btn.classList.add('border-zinc-700', 'text-zinc-500');
                    });
                    this.classList.add('active', 'border-acid-lime', 'text-acid-lime');
                    this.classList.remove('border-zinc-700', 'text-zinc-500');

                    // Filter rows
                    signalRows.forEach(row => {
                        const confidence = row.dataset.confidence;
                        if (filter === 'all' || confidence === filter) {
                            row.style.display = '';
                        } else {
                            row.style.display = 'none';
                        }
                    });
                });
            });
        });
    </script>
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
        
        # --- FIXED PREDICTION FOR BOOSTER ---
        try:
            # Try predicting as XGBRegressor
            raw_pred = -1 * model.predict(X)[0]
        except:
            # Fallback to Booster prediction (requires DMatrix)
            dmat = xgb.DMatrix(X)
            raw_pred = -1 * model.predict(dmat)[0]

        fair_line = max(min(raw_pred, 21), -21)
        
        raw_spread = game['spread_line']
        if pd.isna(raw_spread): continue
        vegas_line = -1 * raw_spread if FIX_VEGAS_SIGNS else raw_spread
        
        diff = fair_line - vegas_line
        if abs(diff) > 10: fair_line = vegas_line + (diff * 0.5) 
        
        raw_edge = round(vegas_line - fair_line, 2)
        abs_edge = abs(raw_edge)
        
        units, conf_badge = calculate_kelly_units(abs_edge)
        
        action = "PASS"; conf = ""; 
        
        if units > 0.0:
            pick_team = game['home_team'] if raw_edge > 0 else game['away_team']
            action = f"BET {pick_team}"
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
                'pick_team': pick,
                'home': game['home_team'], 'away': game['away_team'],
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

        # --- FIXED PREDICTION FOR BOOSTER ---
        try:
            raw_pred = -1 * model.predict(X)[0]
        except:
            dmat = xgb.DMatrix(X)
            raw_pred = -1 * model.predict(dmat)[0]

        fair_line = max(min(raw_pred, 21), -21)
        
        raw_spread = game['spread_line']
        if pd.isna(raw_spread): continue
        vegas_line = -1 * raw_spread if FIX_VEGAS_SIGNS else raw_spread
        
        diff = fair_line - vegas_line
        if abs(diff) > 10: fair_line = vegas_line + (diff * 0.5)
        
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
            'home': game['home_team'], 'away': game['away_team'],
            'Vegas': vegas_dsp, 'Fair_Line': fair_dsp, 
            'Edge': round(abs_edge, 1),
            'Action': action, 
            'Conf': conf, 
            'clean_conf': conf.replace("🔥 ", "").replace("⚠️ ", "").replace("💪 ", "").replace("None", ""),
            'week': next_week, 'game_id': game['game_id'],
            'units': units,
            'pick': action.replace("BET ", "").split(" (")[0], # Just the team name
            'pick_full': action.replace("BET ", "")
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
    
    # Calculate system confidence based on win rate variance
    confidence_level, confidence_color, confidence_detail = calculate_system_confidence(graded)
        
    html = template.render(
        active_bets=active_bets,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        roi=round((profit/total)*100, 1) if total > 0 else 0.0,
        units=round(profit, 2),
        record=f"{wins}-{losses}-{pushes}",
        win_pct=round((wins/total)*100, 1) if total > 0 else 0.0,
        history=recent,
        confidence_level=confidence_level,
        confidence_color=confidence_color,
        confidence_detail=confidence_detail
    )
    
    with open(DOCS_PATH, "w", encoding="utf-8") as f: f.write(html)
    full_path = os.path.abspath(DOCS_PATH)
    print(f"✅ Dashboard Updated: {full_path}")
    webbrowser.open(f"file://{full_path}")
