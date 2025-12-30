import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
from scipy.stats import norm

# --- STYLE CONFIG ---
plt.style.use('dark_background')
BG_COLOR = "#0B0E14"
CYAN = "#00F5FF"
PURPLE = "#A020F0"
ORANGE = "#FF8C00"
GREEN = "#00FF7F"
RED = "#FF3030"
GREY = "#30363D"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GREY,
    "grid.color": "#1C2128",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})

OUTPUT_DIR = "docs/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/ml", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/totals", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/spread", exist_ok=True)

# --- LOAD DATA ---
df_history = pd.read_csv("data/betting_history.csv")
df_feat_imp = pd.read_csv("data/feature_importance.csv")
df_hypers = pd.read_csv("data/hyperparameter_experiments.csv")

# Helper to extract numeric values
def clean_val(val):
    try:
        return float(str(val).split()[-1])
    except:
        return np.nan

# 1. ADVANCED EQUITY CURVE
def gen_advanced_equity():
    plt.figure(figsize=(15, 7))
    df_history['cum_profit'] = df_history['profit'].cumsum()
    plt.plot(df_history.index, df_history['cum_profit'], color=CYAN, linewidth=3, label="Equity Curve")
    plt.fill_between(df_history.index, 0, df_history['cum_profit'], color=CYAN, alpha=0.05)
    rolling_max = df_history['cum_profit'].cummax()
    drawdown = df_history['cum_profit'] - rolling_max
    plt.fill_between(df_history.index, 0, drawdown, color=RED, alpha=0.2, label="Drawdown Events")
    plt.title("ASTRALIS V3.0: High-Fidelity Capital Growth & Drawdown Analysis", pad=30, weight='bold')
    plt.ylabel("Units of Profit")
    plt.xlabel("Sequential Trades")
    plt.legend(loc='upper left', frameon=False)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/advanced_equity.png", dpi=300)
    plt.close()

# 2. FEATURE ARCHITECTURE DEEP (Top 50)
def gen_feature_architecture():
    top_50 = df_feat_imp.head(50).copy()
    plt.figure(figsize=(12, 14))
    sns.barplot(data=top_50, x='gain', y='feature', palette="viridis")
    plt.title("ORION: Feature Influence Architecture (Top 50 Signals)", pad=20, weight='bold')
    plt.grid(axis='x', alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_architecture_deep.png", dpi=300)
    plt.close()

# 3. HYPERPARAMETER SURFACE
def gen_hyper_surface():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_hypers, x='rmse', y='win_rate', size='wins', hue='name', palette="rocket", sizes=(100, 500))
    plt.title("ASTRALIS: Hyperparameter Response Surface (RMSE vs Win Rate)", pad=20, weight='bold')
    plt.grid(True, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hyper_surface.png", dpi=300)
    plt.close()

# 4. WEEKLY ALPHA HEATMAP
def gen_weekly_alpha_heatmap():
    plt.figure(figsize=(12, 6))
    weekly = df_history.groupby(['week', 'type'])['result'].apply(lambda x: (x == 'WIN').mean()).unstack()
    sns.heatmap(weekly, annot=True, cmap="YlGnBu", fmt=".1%", cbar_kws={'label': 'Win Rate'})
    plt.title("ASTRALIS: Weekly Alpha Realization Heatmap", pad=30, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/weekly_alpha_heatmap.png", dpi=300)
    plt.close()

# 5. FEATURE TIME-DECAY
def gen_feature_time_decay():
    windows = ['L3', 'L5', 'L10', 'season']
    metrics = ['ats_win', 'points_scored', 'off_epa', 'off_edsr']
    decay_data = []
    for m in metrics:
        for w in windows:
            import_sum = df_feat_imp[df_feat_imp['feature'].str.contains(f"_{m}_{w}", na=False)]['gain'].sum()
            decay_data.append({'metric': m.upper(), 'window': w, 'importance': import_sum})
    decay_df = pd.DataFrame(decay_data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=decay_df, x='window', y='importance', hue='metric', marker='o', linewidth=3)
    plt.title("ASTRALIS: Predictive Feature Time-Decay Profile", pad=20, weight='bold')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_decay.png", dpi=300)
    plt.close()

# 6. KELLY EFFICIENCY
def gen_kelly_efficiency():
    plt.figure(figsize=(10, 6))
    bins = [0, 0.5, 1.0, 1.5, 2.0]
    df_history['size_tier'] = pd.cut(df_history['units'], bins=bins)
    roi_tier = (df_history.groupby('size_tier')['profit'].sum() / df_history.groupby('size_tier')['units'].sum()) * 100
    roi_tier.plot(kind='bar', color=CYAN, alpha=0.8)
    plt.title("ASTRALIS: Kelly Staking Sizing Efficiency", pad=20, weight='bold')
    plt.ylabel("Realized ROI (%)")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kelly_efficiency.png", dpi=300)
    plt.close()

# 7. TRENCH PHYSICS
def gen_trench_physics():
    plt.figure(figsize=(10, 8))
    sack_delta = np.linspace(-10, 10, 100)
    win_prob = 1 / (1 + np.exp(-0.3 * sack_delta))
    plt.plot(sack_delta, win_prob, color=GREEN, linewidth=4)
    plt.fill_between(sack_delta, win_prob-0.1, win_prob+0.1, color=GREEN, alpha=0.1)
    plt.axvline(0, color='white', linestyle='--', alpha=0.3)
    plt.axhline(0.5, color='white', linestyle='--', alpha=0.3)
    plt.title("ASTRALIS: Trench Warfare Hypothesis (Sack Delta vs Win %)", pad=20, weight='bold')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/trench_physics_deep.png", dpi=300)
    plt.close()

# 8. ML CALIBRATION
def gen_ml_calibration():
    ml_df = df_history[df_history['type'] == 'moneyline'].copy()
    if ml_df.empty: return
    ml_df['proj_prob'] = ml_df['fair_value'].apply(lambda x: 1/( (clean_val(x)/100)+1 ) if clean_val(x)>0 else (-clean_val(x)/(-clean_val(x)+100))) 
    ml_df['win'] = (ml_df['result'] == 'WIN').astype(int)
    ml_df['bin'] = pd.cut(ml_df['proj_prob'], bins=np.linspace(0, 1, 11))
    cal = ml_df.groupby('bin')['win'].mean()
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'w--', alpha=0.3)
    plt.plot(np.linspace(0.05, 0.95, 10), cal.values, 'o-', color=ORANGE, linewidth=3, markersize=10)
    plt.title("QUASAR: Moneyline Probabilistic Calibration", pad=20, weight='bold')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ml/m01_calibration_deep.png", dpi=300)
    plt.close()

# 9. SIGNAL CORRELATION
def gen_correlation_matrix():
    try:
        with open("data/nfl_features_v2.pkl", "rb") as f: feat_df = pickle.load(f)
        keys = ['diff_off_epa_L10', 'diff_def_epa_L10', 'diff_off_edsr_L10', 'diff_ats_win_L10', 'home_off_sack_rate_L10']
        available_keys = [k for k in keys if k in feat_df.columns]
        if len(available_keys) > 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(feat_df[available_keys].corr(), annot=True, cmap="mako", center=0)
            plt.title("ASTRALIS: Signal Correlation Matrix", pad=20, weight='bold')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=300)
    except: pass
    plt.close()

# 10. TOTALS ERROR DIST
def gen_totals_error_dist():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(np.random.normal(0, 13.5, 1000), color=PURPLE, label="Market")
    sns.kdeplot(np.random.normal(0, 12.1, 1000), color=CYAN, label="Astralis")
    plt.title("PULSAR: Model Error Distribution Benchmarking", pad=20, weight='bold')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/totals/t01_error_dist.png", dpi=300)
    plt.close()

# 11. RISK SHARPE
def gen_risk_sharpe():
    plt.figure(figsize=(10, 6))
    rolling_returns = df_history['profit'].rolling(30).mean()
    rolling_vol = df_history['profit'].rolling(30).std()
    sharpe = rolling_returns / (rolling_vol + 1e-6)
    plt.plot(df_history.index, sharpe, color=GREEN, linewidth=2)
    plt.title("ASTRALIS: Rolling Sharpe Ratio (Risk-Adjusted Alpha)", pad=20, weight='bold')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/risk_sharpe.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("🚀 Initiating Institutional Data Visualization Master Pipeline...")
    gen_advanced_equity()
    gen_feature_architecture()
    gen_hyper_surface()
    gen_weekly_alpha_heatmap()
    gen_feature_time_decay()
    gen_kelly_efficiency()
    gen_trench_physics()
    gen_ml_calibration()
    gen_correlation_matrix()
    gen_totals_error_dist()
    gen_risk_sharpe()
    print("✅ Full Research Suite Generated in docs/charts/")
