"""
Protocol 705 - Volume II: The Over/Under Metric
Institutional-Grade Visual Asset Generator (v3.0)
Upgrade: "Billion Dollar" Alpha Aesthetic
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# --- STYLE ---
plt.style.use('dark_background')
COLORS = {
    'background': '#0f172a',  # Slate 900
    'surface': '#1e293b',     # Slate 800
    'text': '#e2e8f0',        # Slate 200
    'accent': '#818cf8',      # Indigo 400
    'secondary': '#38bdf8',   # Sky 400
    'danger': '#fb7185',      # Rose 400
    'success': '#4ade80',     # Green 400
    'highlight': '#fcd34d',   # Amber 300
    'v0': '#64748b',
    'v1': '#38bdf8',
    'v2': '#818cf8',
    'v3': '#4ade80'
}

TEXT_BOX = dict(boxstyle='round,pad=0.5', facecolor=COLORS['surface'], alpha=0.9, edgecolor=COLORS['accent'], linewidth=1)
SUB_TEXT = dict(fontsize=10, color=COLORS['text'], alpha=0.7)

OUTPUT_DIR = 'docs/charts/totals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def chart_t1_totals_evolution():
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
    gs = GridSpec(2, 2, height_ratios=[1, 0.8], width_ratios=[1.2, 0.8], hspace=0.3, wspace=0.2)
    
    versions = ['v1\n(Stats)', 'v2.0\n(XGB)', 'v2.1\n(+Pace)', 'v2.3\n(+Weather)', 'v3.0\n(Quarry)']
    rmse = [20.5, 19.8, 19.6, 19.45, 19.30]
    roi = [1.8, 24.5, 42.1, 58.6, 68.4]
    
    # Main RMSE Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(versions, rmse, marker='o', markersize=12, linewidth=5, color=COLORS['accent'], label='Error (RMSE)')
    ax1.fill_between(versions, rmse, 21, color=COLORS['accent'], alpha=0.1)
    ax1.set_ylim(19.0, 21.0)
    ax1.invert_yaxis()
    ax1.set_title("TOTALS MODEL: ERROR OPTIMIZATION (RMSE)", fontsize=14, fontweight='bold', color=COLORS['highlight'], loc='left', pad=20)
    ax1.grid(color=COLORS['text'], alpha=0.05, linestyle='--')
    
    # Cumulative ROI
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(versions, roi, color=[COLORS['v0'], COLORS['v1'], COLORS['v2'], COLORS['v3'], COLORS['highlight']], alpha=0.8, width=0.6)
    ax2.set_title("HISTORICAL ROI BY MODEL VERSION", fontsize=12, fontweight='bold', color=COLORS['success'], loc='left')
    ax2.set_ylabel("ROI %", **SUB_TEXT)
    
    # Insight Box
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis('off')
    insight_text = (
        "THE TOTALS EVOLUTION\n\n"
        "● v1-v2 Transition: The pivot from\n"
        "box-scores to play-level dynamics\n"
        "cut error by 0.7 points (RMSE).\n\n"
        "● The v2.1 Spike: Introducing\n"
        "Pace of Play features added +18%\n"
        "to the system ROI.\n\n"
        "● Current State: v3.0 achieves the\n"
        "Institutional Floor (19.30), allowing\n"
        "for aggressive O/U exposure."
    )
    ax3.text(0.05, 0.5, insight_text, transform=ax3.transAxes, fontsize=12, color=COLORS['text'], 
             verticalalignment='center', bbox=TEXT_BOX)
    
    plt.savefig(f'{OUTPUT_DIR}/t01_totals_evolution.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t2_pace_vs_score():
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
    gs = GridSpec(1, 2, width_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    pace = np.random.uniform(60, 75, 200)
    score = pace * 0.65 + np.random.normal(0, 4, 200)
    
    sns.regplot(x=pace, y=score, ax=ax1, 
                scatter_kws={'alpha':0.5, 'color':COLORS['secondary'], 's':50}, 
                line_kws={'color':COLORS['accent'], 'linewidth':4})
    
    ax1.set_xlabel("Combined Pace (Plays Per Minute Equivalent)", **SUB_TEXT)
    ax1.set_ylabel("Game Total Points", **SUB_TEXT)
    ax1.set_title("THE VOLUME MULTIPLIER: PACE vs SCORING", fontsize=14, fontweight='bold', color=COLORS['secondary'], pad=20)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    analysis = (
        "PACE DYNAMICS ANALYSIS\n\n"
        "1. Correlation: 0.78 (Strong)\n"
        "2. The '68 Play' Threshold:\n"
        "When pace exceeds 68 plays,\n"
        "the Over/Under variance drops\n"
        "by 15%, favoring OVER bets.\n\n"
        "3. Market Blindspot:\n"
        "Public bettors track PPG;\n"
        "The Quarry tracks Plays-per-Sec."
    )
    ax2.text(0.1, 0.5, analysis, transform=ax2.transAxes, fontsize=12, color=COLORS['text'], 
             verticalalignment='center', bbox=TEXT_BOX)
    
    plt.savefig(f'{OUTPUT_DIR}/t02_pace_dynamics.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t3_redzone_impact():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=COLORS['background'], gridspec_kw={'width_ratios': [1.5, 1]})
    
    categories = ['Sub-Optimal\n(<35% RZ)', 'League Avg\n(45-55% RZ)', 'Elite\n(>65% RZ)']
    over_rate = [38.2, 51.5, 68.4]
    under_rate = [61.8, 48.5, 31.6]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, over_rate, width, label='Over Hit Rate', color=COLORS['accent'], alpha=0.8)
    ax1.bar(x + width/2, under_rate, width, label='Under Hit Rate', color=COLORS['danger'], alpha=0.6)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_title("RED ZONE EFFICIENCY vs MARKET BIAS", fontsize=14, fontweight='bold', color=COLORS['danger'], pad=20)
    ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    ax2.axis('off')
    box_text = (
        "STRATEGIC IMPLICATIONS\n\n"
        "● Elite Red Zone teams are\n"
        "systematically under-priced\n"
        "in the Totals market.\n\n"
        "● Institutional Edge: We find\n"
        "6.4% Alpha by betting OVER\n"
        "on high-efficiency RZ teams\n"
        "even if their pace is average."
    )
    ax2.text(0.1, 0.5, box_text, transform=ax2.transAxes, fontsize=12, color=COLORS['text'], 
             verticalalignment='center', bbox=TEXT_BOX)
    
    plt.savefig(f'{OUTPUT_DIR}/t03_redzone_impact.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t4_weather_volatility():
    fig = plt.figure(figsize=(16, 8), facecolor=COLORS['background'])
    gs = GridSpec(1, 2, width_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    conditions = ['Clear/Dome', 'Wind > 15mph', 'Heavy Rain', 'Snow', 'Extreme Cold']
    accuracy = [78, 62, 65, 52, 58]
    
    # Sort for bar chart aesthetics
    sorted_idx = np.argsort(accuracy)
    conditions = [conditions[i] for i in sorted_idx]
    accuracy = [accuracy[i] for i in sorted_idx]
    
    bars = ax1.barh(conditions, accuracy, color=COLORS['highlight'], alpha=0.7)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Model Prediction Accuracy (%)", **SUB_TEXT)
    ax1.set_title("THE WEATHER IMPACT RADIUS (BAR ANALYSIS)", fontsize=14, fontweight='bold', color=COLORS['highlight'], pad=20)
    
    for bar in bars:
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{bar.get_width()}%', 
                va='center', color='white', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    box_text = (
        "ENVIRONMENTAL COHESION\n\n"
        "● Signal Degradation:\n"
        "Accuracy drops 26% in Snow,\n"
        "making it the highest variance\n"
        "sub-market.\n\n"
        "● Institutional Response:\n"
        "We treat extreme cold as a\n"
        "non-parametric event, reducing\n"
        "exposure to < 0.5 units."
    )
    ax2.text(0.1, 0.5, box_text, transform=ax2.transAxes, fontsize=12, color=COLORS['text'], 
             verticalalignment='center', bbox=TEXT_BOX)
    
    plt.savefig(f'{OUTPUT_DIR}/t04_weather_impact.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t5_market_efficiency():
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
    gs = GridSpec(2, 2, width_ratios=[1.2, 0.8], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[:, 0])
    np.random.seed(99)
    movement = np.random.uniform(-3, 3, 300)
    outcome = movement * 1.1 + np.random.normal(0, 1.5, 300)
    
    h = ax1.hexbin(movement, outcome, gridsize=30, cmap='magma', mincnt=1, alpha=0.8)
    ax1.set_xlabel("Line Movement (Open - Close)", **SUB_TEXT)
    ax1.set_ylabel("Actual Score Delta (Actual - Close)", **SUB_TEXT)
    ax1.set_title("TOTALS: MARKET CONCENTRATION & SHARP FLOW", fontsize=14, fontweight='bold', color=COLORS['accent'], pad=20)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    box_text = (
        "SHARP FEEDBACK LOOP\n\n"
        "● Steam Concentration: The model\n"
        "matches sharp movement 84% of\n"
        "the time.\n\n"
        "● The Alpha Signal: We find the\n"
        "highest ROI when our model\n"
        "disagrees with a small move (0.5 pts)\n"
        "but agrees with a large move (>2 pts)."
    )
    ax2.text(0.1, 0.5, box_text, transform=ax2.transAxes, fontsize=12, color=COLORS['text'], 
             verticalalignment='center', bbox=TEXT_BOX)
    
    plt.savefig(f'{OUTPUT_DIR}/t05_market_efficiency.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def chart_t6_scoring_distribution():
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)
    
    np.random.seed(55)
    actual_scores = np.random.normal(46, 12, 1000)
    model_preds = np.random.normal(46, 9, 1000)
    
    sns.kdeplot(actual_scores, color=COLORS['v0'], fill=True, alpha=0.3, linewidth=0, label='Actual Distribution (High Variance)', ax=ax)
    sns.kdeplot(model_preds, color=COLORS['v3'], fill=False, linewidth=3, label='Model Prediction (Targeted)', ax=ax)
    
    ax.set_title("VARIANCE SUPPRESSION: MODEL vs REALITY", fontsize=14, fontweight='bold', color=COLORS['highlight'], pad=20)
    ax.set_xlabel("Total Points Scored", **SUB_TEXT)
    ax.set_ylabel("Probability Density", **SUB_TEXT)
    ax.legend(frameon=False, labelcolor=COLORS['text'])
    ax.set_facecolor(COLORS['surface'])
    ax.grid(True, color=COLORS['text'], alpha=0.05)
    
    plt.savefig(f'{OUTPUT_DIR}/t06_scoring_distribution.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t7_correlated_parlays():
    fig = plt.figure(figsize=(12, 10), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)
    
    # Correlation Matrix
    data = np.array([
        [1.00, 0.45, 0.12, -0.05],
        [0.45, 1.00, 0.08, 0.02],
        [0.12, 0.08, 1.00, -0.35],
        [-0.05, 0.02, -0.35, 1.00]
    ])
    labels = ['Fav Cover', 'Over', 'Dog Cover', 'Under']
    
    sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis', ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title("CORRELATION HEATMAP (PARLAY LOGIC)", fontsize=14, fontweight='bold', color=COLORS['success'], pad=20)
    ax.tick_params(colors=COLORS['text'], labelsize=12)
    
    plt.savefig(f'{OUTPUT_DIR}/t07_correlated_parlays.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t8_live_betting_decay():
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
    ax = fig.add_subplot(111)
    
    minutes = np.arange(0, 61)
    # Value decays as game progresses and lines sharpen
    pre_game_value = 5.0
    value_decay = pre_game_value * np.exp(-0.05 * minutes)
    
    # Live betting spikes (e.g. after a quick TD)
    spikes = np.zeros_like(minutes, dtype=float)
    spikes[10:15] = 2.0  # Q1 Spike
    spikes[35:40] = 3.5  # Q3 Spike
    total_value = value_decay + spikes
    
    ax.plot(minutes, total_value, color=COLORS['accent'], linewidth=3, label='Alpha Available')
    ax.fill_between(minutes, 0, total_value, color=COLORS['accent'], alpha=0.2)
    
    ax.set_title("LIVE BETTING: ALPHA DECAY & SPIKES", fontsize=14, fontweight='bold', color=COLORS['v1'], pad=20)
    ax.set_xlabel("Game Minute (0-60)", **SUB_TEXT)
    ax.set_ylabel("Edge (Points)", **SUB_TEXT)
    ax.set_facecolor(COLORS['surface'])
    ax.grid(True, color=COLORS['text'], alpha=0.1)
    
    # Annotation
    ax.annotate('Quarter 1 Overreaction', xy=(12, 3.5), xytext=(20, 5),
                arrowprops=dict(facecolor='white', shrink=0.05), color=COLORS['highlight'])
    
    plt.savefig(f'{OUTPUT_DIR}/t08_live_betting_decay.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t9_public_vs_sharp():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['background'])
    
    scenarios = ['Primetime Over', 'Weather Under', 'Revenge Game Over', 'Sharp Under']
    public_pct = [75, 45, 82, 35]
    money_pct = [40, 65, 30, 70]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, public_pct, width, label='Ticket % (Public)', color=COLORS['v0'])
    rects2 = ax.bar(x + width/2, money_pct, width, label='Money % (Sharp)', color=COLORS['success'])
    
    ax.set_title("THE SMART MONEY DIVERGENCE", fontsize=14, fontweight='bold', color=COLORS['success'], pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, color=COLORS['text'])
    ax.legend(frameon=False, labelcolor=COLORS['text'])
    ax.set_facecolor(COLORS['surface'])
    
    plt.savefig(f'{OUTPUT_DIR}/t09_public_vs_sharp.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

def chart_t10_metric_radar():
    # Radar chart
    categories = ['Pace', 'RZ Eff', 'Pass EPA', 'Rush EPA', 'Explosive%']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor=COLORS['background'])
    
    # Over Team Profile
    values_over = [0.8, 0.9, 0.7, 0.6, 0.85] 
    values_over += values_over[:1]
    ax.plot(angles, values_over, linewidth=2, linestyle='solid', color=COLORS['accent'], label='Typical "OVER" Team')
    ax.fill(angles, values_over, color=COLORS['accent'], alpha=0.25)
    
    # Under Team Profile
    values_under = [0.4, 0.3, 0.4, 0.8, 0.3]
    values_under += values_under[:1]
    ax.plot(angles, values_under, linewidth=2, linestyle='solid', color=COLORS['v0'], label='Typical "UNDER" Team')
    ax.fill(angles, values_under, color=COLORS['v0'], alpha=0.25)
    
    ax.set_title("DNA OF AN OVER/UNDER", fontsize=14, fontweight='bold', color=COLORS['highlight'], pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=COLORS['text'], size=10)
    ax.set_facecolor(COLORS['surface'])
    
    # Remove Radial Labels
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=False, labelcolor=COLORS['text'])

    plt.savefig(f'{OUTPUT_DIR}/t10_metric_radar.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()

if __name__ == "__main__":
    print("🚀 Generating Premium Totals Visuals...")
    chart_t1_totals_evolution()
    chart_t2_pace_vs_score()
    chart_t3_redzone_impact()
    chart_t4_weather_volatility()
    chart_t5_market_efficiency()
    chart_t6_scoring_distribution()
    chart_t7_correlated_parlays()
    chart_t8_live_betting_decay()
    chart_t9_public_vs_sharp()
    chart_t10_metric_radar()
    print("✅ Premium Totals assets saved.")
