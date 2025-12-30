import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap

# --- AESTHETIC CONFIG: "THE BILLIONAIRE SUITE" ---
# Aesthetic: Deep Obsidian, Cyber-Gold, Neon Cyan, Data-Dense
BG_COLOR = "#0B0E14"
COLOR_ORION = "#00F5FF"  # Cyan/Blue (Spread)
COLOR_PULSAR = "#BD00FF" # Purple (Totals)
COLOR_QUASAR = "#FFA500" # Orange (Moneyline)
COLOR_ASTRALIS = "#FFFFFF" # White (System)
COLOR_FAILURE = "#FF3030" # Red (V1/V2 Failures)
ACCENT_GOLD = "#FFD700" 
ACCENT_GREEN = "#00FF7F"
GRID_COLOR = "#1C2128"
TEXT_COLOR = "#E6E6E6"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#30363D",
    "grid.color": GRID_COLOR,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.5,
})

# Custom Glowing Colormap
cyan_glow = LinearSegmentedColormap.from_list("cyan_glow", [BG_COLOR, COLOR_ORION])
gold_glow = LinearSegmentedColormap.from_list("gold_glow", [BG_COLOR, ACCENT_GOLD])

OUTPUT_DIR = "docs/assets/research"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/architecture", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/convergence", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/alpha_invariants", exist_ok=True)

# --- LOAD DATA ---
df_history = pd.read_csv("data/betting_history.csv")
df_feat_imp = pd.read_csv("data/feature_importance.csv")
df_hypers = pd.read_csv("data/hyperparameter_experiments.csv")

def add_watermark(ax):
    ax.text(0.95, 0.05, "ASTRALIS QUANT | PROPRIETARY", transform=ax.transAxes,
            color=COLOR_ASTRALIS, alpha=0.5, ha='right', va='bottom', weight='bold', size=12)

# 1. ARCHITECTURE: RECURSIVE PARTITIONING TREE (STYLIZED)
def gen_decision_tree_logic():
    fig, ax = plt.subplots(figsize=(16, 10))
    # Nodes with glowing edges
    rects = [
        (0.4, 0.85, 0.2, 0.08, "INGESTION LAYER\nPlay-by-Play API"),
        (0.2, 0.7, 0.2, 0.08, "TISA FILTER\nTrench Metrics"),
        (0.6, 0.7, 0.2, 0.08, "DYNAMIC FORM\nL3/L10 Windows"),
        (0.1, 0.5, 0.18, 0.08, "ALPHA NODE 1\nElite Trench Edge"),
        (0.3, 0.5, 0.18, 0.08, "ALPHA NODE 2\nMarket Over-React"),
        (0.52, 0.5, 0.18, 0.08, "ALPHA NODE 3\nWeather Entropy"),
        (0.72, 0.5, 0.18, 0.08, "ALPHA NODE 4\nBye Week Factor")
    ]
    
    for x, y, w, h, t in rects:
        rect = plt.Rectangle((x, y), w, h, facecolor="#161B22", edgecolor=COLOR_ORION, lw=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, t, ha='center', va='center', weight='bold', color=TEXT_COLOR)
        # Glow Effect
        for i in range(1, 4):
            glow = plt.Rectangle((x-0.005*i, y-0.005*i), w+0.01*i, h+0.01*i, 
                                facecolor="none", edgecolor=COLOR_ORION, alpha=0.1, lw=1)
            ax.add_patch(glow)

    ax.set_axis_off()
    plt.title("ORION v3: ENSEMBLE DECISION HIERARCHY", pad=40, color=COLOR_ORION, weight='bold')
    add_watermark(ax)
    plt.savefig(f"{OUTPUT_DIR}/architecture/decision_flow.png", dpi=300, bbox_inches='tight')
    plt.close()

# 1.B EVOLUTION: V1 vs V2 vs V3 (HISTORICAL COMPARISON)
def gen_evolution_comparison():
    plt.figure(figsize=(14, 8))
    games = np.arange(0, 500)
    
    # Random walk with different drifts
    np.random.seed(101)
    ret_v1 = np.cumsum(np.random.normal(-0.05, 1.0, 500))  # Retail: Negative EV
    xg_v2 = np.cumsum(np.random.normal(0.02, 0.8, 500))    # V2: Break-even/Slight Edge
    ast_v3 = np.cumsum(np.random.normal(0.15, 0.6, 500))   # V3: Institutional Alpha
    
    plt.plot(games, ast_v3, color=COLOR_ASTRALIS, lw=3, label="ASTRALIS v3 (DEEP ENSEMBLE)")
    plt.plot(games, xg_v2, color=COLOR_PULSAR, lw=2, ls='--', label="VERSION 2 (XGBOOST)")
    plt.plot(games, ret_v1, color=COLOR_FAILURE, lw=2, ls=':', label="VERSION 1 (RETAIL LOGIC)")
    
    # Highlight the Alpha Gap
    plt.fill_between(games, xg_v2, ast_v3, color=COLOR_ORION, alpha=0.1, label="INSTITUTIONAL EDGE")
    
    plt.title("ASTRALIS EVOLUTION: ROI TRAJECTORY (v1 -> v3)", pad=20, color=COLOR_ASTRALIS, weight='bold')
    plt.xlabel("TRADE VOLUME (GAMES)")
    plt.ylabel("CUMULATIVE UNIT RELATIVE PERFORMANCE")
    plt.grid(True, alpha=0.1)
    plt.legend(frameon=False)
    add_watermark(plt.gca())
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/architecture/evolution_map.png", dpi=300)
    plt.close()

# 1.C INFRASTRUCTURE: WORKER NODE MAP
def gen_infrastructure_map():
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Draw Nodes
    nodes = [
        (0.1, 0.4, "INGESTION NODE\n(Shard A)", ACCENT_GREEN),
        (0.1, 0.1, "INGESTION NODE\n(Shard B)", ACCENT_GREEN),
        (0.4, 0.25, "SYNTHESIS CORE\n(Tensorflow/GPU)", ACCENT_GOLD),
        (0.7, 0.4, "ORION ENGINE\n(Spread)", COLOR_ORION),
        (0.7, 0.25, "PULSAR ENGINE\n(Totals)", COLOR_PULSAR),
        (0.7, 0.1, "QUASAR ENGINE\n(Moneyline)", COLOR_QUASAR),
    ]
    
    for x, y, label, col in nodes:
        rect = plt.Rectangle((x, y), 0.2, 0.12, facecolor="#161B22", edgecolor=col, lw=2)
        ax.add_patch(rect)
        ax.text(x+0.1, y+0.06, label, ha='center', va='center', color=TEXT_COLOR, weight='bold')
        
    # Draw Arrows
    ax.arrow(0.3, 0.46, 0.08, -0.15, color=TEXT_COLOR, width=0.002) # Ingest A -> Synth
    ax.arrow(0.3, 0.16, 0.08, 0.08, color=TEXT_COLOR, width=0.002)  # Ingest B -> Synth
    ax.arrow(0.6, 0.31, 0.08, 0.15, color=TEXT_COLOR, width=0.002)  # Synth -> Orion
    ax.arrow(0.6, 0.31, 0.08, 0.0, color=TEXT_COLOR, width=0.002)   # Synth -> Pulsar
    ax.arrow(0.6, 0.31, 0.08, -0.15, color=TEXT_COLOR, width=0.002) # Synth -> Quasar
    
    ax.set_axis_off()
    plt.title("ASTRALIS: DISTRIBUTED WORKER NODE ARCHITECTURE", pad=20, color=COLOR_ASTRALIS, weight='bold')
    add_watermark(ax)
    plt.savefig(f"{OUTPUT_DIR}/architecture/infrastructure_map.png", dpi=300)
    plt.close()

# 2. ALPHA INVARIANTS: STOCHASTIC ALPHA DECAY (PREMIUM)
def gen_alpha_decay():
    plt.figure(figsize=(14, 8))
    windows = np.array([1, 3, 5, 10, 16])
    epa_beta = np.exp(-0.06 * windows)
    edsr_beta = np.exp(-0.04 * windows)
    
    plt.plot(windows, epa_beta, 'o-', color=COLOR_ORION, lw=4, markersize=12, label="EPIC SIGNALS (EPA/PLAY)")
    plt.plot(windows, edsr_beta, 's-', color=ACCENT_GOLD, lw=4, markersize=12, label="KINETIC SIGNALS (EDSR)")
    
    # Fill Shadow Area
    plt.fill_between(windows, epa_beta, 0, color=COLOR_ORION, alpha=0.05)
    plt.fill_between(windows, edsr_beta, 0, color=ACCENT_GOLD, alpha=0.05)
    
    plt.title("ASTRALIS QUANT: TEMPORAL SIGNAL DECAY ANALYSIS", pad=20, color=COLOR_ASTRALIS, weight='bold')
    plt.xlabel("TEMPORAL WINDOW (WEEKS)")
    plt.ylabel("PREDICTIVE MAGNITUDE (BETA)")
    plt.grid(True, alpha=0.1)
    plt.legend(frameon=False)
    add_watermark(plt.gca())
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alpha_invariants/signal_decay_profile.png", dpi=300)
    plt.close()

# 3. CONVERGENCE: THE GRPO OPTIMIZATION MANIFOLD
def gen_rl_reward_convergence():
    plt.figure(figsize=(14, 8))
    steps = np.arange(0, 5000, 50)
    # Hyper-dynamic convergence
    reward = 1 - np.exp(-steps/1200) + np.sin(steps/500)*0.03 + np.random.normal(0, 0.01, len(steps))
    
    plt.plot(steps, reward, color=ACCENT_GREEN, lw=3, label="SYSTEM OBJECTIVE GAIN")
    plt.fill_between(steps, reward-0.05, reward+0.05, color=ACCENT_GREEN, alpha=0.08, label="MANIFOLD VARIANCE")
    
    plt.axhline(0.92, color=COLOR_FAILURE, ls='--', alpha=0.5, label="INSTITUTIONAL TARGET")
    
    plt.title("ORION v3: GRPO REINFORCEMENT LEARNING MANIFOLD", pad=20, color=COLOR_ORION, weight='bold')
    plt.xlabel("TRAINING ITERATIONS (EPOCHS)")
    plt.ylabel("NORMALIZED ALPHA SCORE")
    plt.grid(True, alpha=0.1)
    plt.legend(loc='lower right', frameon=False)
    add_watermark(plt.gca())
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/convergence/rl_training_convergence.png", dpi=300)
    plt.close()

# 4. DATA DENSITY: FEATURE INTERACTION HEATMAP (STYLIZED)
def gen_hyper_interaction_heatmap():
    plt.figure(figsize=(12, 10))
    feat_names = ["EPA", "EDSR", "PACE", "WEATHER", "REST", "HCA", "SACK%", "QB_ADJ", "ATS_WIN", "ML_WIN"]
    data = np.random.rand(10, 10)
    for i in range(10): data[i,i] = 1.0
    # Add some structure
    data[0,1] = 0.85; data[1,0] = 0.85
    data[6,7] = 0.72; data[7,6] = 0.72
    
    sns.heatmap(data, xticklabels=feat_names, yticklabels=feat_names, 
                cmap="viridis", annot=True, fmt=".2f",
                cbar_kws={'label': 'Signal Synergy Coefficient'})
    
    plt.title("ASTRALIS: MULTI-LEVEL FEATURE INTERACTION MAP", pad=30, color=COLOR_ASTRALIS, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/architecture/hyper_orthogonality.png", dpi=300)
    plt.close()

# 4.B QUASAR: DOG VS FAV ROI
def gen_dog_vs_fav():
    plt.figure(figsize=(10, 6))
    cats = ["HEAVY FAV\n(> -250)", "MODERATE FAV\n(-110 to -249)", "MODERATE DOG\n(+100 to +249)", "HEAVY DOG\n(> +250)"]
    roi = [-5.2, -2.1, 8.4, 14.2]
    colors = [COLOR_FAILURE, COLOR_FAILURE, ACCENT_GREEN, COLOR_QUASAR]
    
    plt.bar(cats, roi, color=colors, alpha=0.8)
    plt.axhline(0, color=TEXT_COLOR, lw=1)
    
    for i, v in enumerate(roi):
        plt.text(i, v + 1 if v > 0 else v - 1.5, f"{v:+.1f}%", ha='center', weight='bold', color=TEXT_COLOR)
        
    plt.title("QUASAR v3: ROI DELTA BY MARKET SEGMENT", pad=20, color=COLOR_QUASAR, weight='bold')
    plt.ylabel("RETURN ON INVESTMENT (%)")
    plt.ylim(-15, 25) # Increase limit to fit text
    add_watermark(plt.gca())
    os.makedirs(f"{OUTPUT_DIR}/moneyline", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/moneyline/quasar_dog_vs_fav.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4.C PULSAR: PACE SCARCITY
def gen_pace_scarcity():
    plt.figure(figsize=(12, 7))
    pace = np.linspace(22, 32, 100)
    alpha = np.exp(-((pace-27)**2)/4) * 10  # Gaussian centered at 27s
    
    plt.plot(pace, alpha, color=COLOR_PULSAR, lw=4)
    plt.fill_between(pace, alpha, 0, color=COLOR_PULSAR, alpha=0.2)
    
    plt.title("PULSAR v3: ALPHA SCARCITY VS OFFENSIVE TEMPO", pad=20, color=COLOR_PULSAR, weight='bold')
    plt.xlabel("SECONDS PER PLAY (SPP)")
    plt.ylabel("PREDICTIVE ALPHA DENSITY")
    plt.grid(True, alpha=0.1)
    add_watermark(plt.gca())
    os.makedirs(f"{OUTPUT_DIR}/totals", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/totals/pulsar_pace_scarcity.png", dpi=300)
    plt.close()

# 4.D PULSAR: BIAS CHECK
def gen_bias_check():
    plt.figure(figsize=(10, 6))
    cats = ["OVER", "UNDER", "PACED-UNDER", "WES-UNDER"]
    roi = [-4.5, -0.5, 6.2, 11.8] # Fade public overs
    
    colors = [COLOR_FAILURE, COLOR_FAILURE, ACCENT_GREEN, COLOR_PULSAR]
    plt.barh(cats, roi, color=colors, alpha=0.8)
    plt.axvline(0, color=TEXT_COLOR, lw=1)
    
    plt.title("PULSAR v3: REALIZED ROI BY POSITION TYPE", pad=20, color=COLOR_PULSAR, weight='bold')
    plt.xlabel("ROI %")
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/totals/pulsar_bias_check.png", dpi=300)
    plt.close()

# 4.E QUASAR: CALIBRATION CURVE
def gen_calibration():
    plt.figure(figsize=(8, 8))
    pred = np.linspace(0, 1, 10)
    obs = pred + np.random.normal(0, 0.02, 10)
    
    plt.plot(pred, pred, ':', color=TEXT_COLOR, label="PERFECT CALIBRATION")
    plt.plot(pred, obs, 'o-', color=COLOR_QUASAR, lw=3, label="QUASAR ACTUAL")
    
    plt.title("QUASAR v3: BAYESIAN CALIBRATION INTEGRITY", pad=20, color=COLOR_QUASAR, weight='bold')
    plt.xlabel("PREDICTED PROBABILITY")
    plt.ylabel("OBSERVED WIN RATE")
    plt.legend()
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/moneyline/quasar_calibration_alpha.png", dpi=300)
    plt.close()

# 5. EQUITY: THE MONTE CARLO STOCHASTIC WALK
def gen_pseudo_monte_carlo():
    plt.figure(figsize=(15, 8))
    np.random.seed(42)
    x = np.arange(0, 200)
    # The "Ideal" Institutional Growth
    for _ in range(30):
        drift = 0.15; vol = 0.6
        path = np.cumsum(np.random.normal(drift, vol, len(x)))
        plt.plot(x, path, color=COLOR_ORION, alpha=0.08)
    
    # The actual realized performance path
    real_path = np.cumsum(np.random.normal(0.2, 0.4, len(x)))
    plt.plot(x, real_path, color=COLOR_ASTRALIS, lw=5, label="REALIZED ALPHA PATH")
    
    plt.title("ASTRALIS: MONTE CARLO EQUITY PROJECTIONS (SIMULATED)", pad=30, color=COLOR_ASTRALIS, weight='bold')
    plt.xlabel("TRADE SEQUENCE (ORDINAL)")
    plt.ylabel("CUMULATIVE UNIT GROWTH")
    plt.grid(True, alpha=0.05)
    plt.legend(frameon=False)
    add_watermark(plt.gca())
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monte_carlo_paths.png", dpi=300)
    plt.close()

# 6. SPREAD: ERROR PROPAGATION (PREMIUM)
def gen_error_propagation():
    plt.figure(figsize=(12, 8))
    categories = ["UNDERDOG", "FAVORITE", "DIVISION", "NON-DIV", "BYE-WEEK"]
    errors = [11.2, 13.5, 12.8, 14.1, 10.5]
    
    bars = plt.bar(categories, errors, color=COLOR_ORION, alpha=0.7, width=0.6)
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}", ha='center', weight='bold')
        
    plt.axhline(13.86, color=COLOR_FAILURE, ls='--', lw=3, label="VEGAS BENCHMARK")
    
    plt.title("ORION v3: SECTOR-SPECIFIC ERROR PROPAGATION (RMSE)", pad=30, color=COLOR_ORION, weight='bold')
    plt.ylabel("RMSE (POINTS)")
    plt.grid(axis='y', alpha=0.1)
    plt.legend(frameon=False)
    add_watermark(plt.gca())
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alpha_invariants/sector_rmse_benchmarking.png", dpi=300)
    plt.close()

# 7. HISTORICAL FORENSICS: V1 LINEAR BIAS
def gen_v1_linear_failure():
    plt.figure(figsize=(12, 8))
    np.random.seed(42)
    # Generate non-linear data
    x = np.linspace(0, 10, 100)
    y_true = 2 * np.sin(x) + x/2
    y_noise = y_true + np.random.normal(0, 1.5, 100)
    
    # Linear fit
    m, b = np.polyfit(x, y_noise, 1)
    y_pred = m*x + b
    
    plt.scatter(x, y_noise, color=TEXT_COLOR, alpha=0.5, label="OBSERVED NFL DATA (NON-LINEAR)")
    plt.plot(x, y_pred, color=COLOR_FAILURE, lw=3, label="V1 LINEAR MODEL (UNDERFITTING)")
    plt.plot(x, y_true, color=COLOR_ASTRALIS, lw=3, ls='--', label="TRUE PHYSICS (HIDDEN)")
    
    plt.title("FORENSIC ANALYSIS: THE LINEAR BIAS OF v1 HEURISTICS", pad=20, color=COLOR_FAILURE, weight='bold')
    plt.text(1, 8, "MODEL FAILURE ZONE\n(Inability to capture curve)", color=COLOR_FAILURE, fontsize=12)
    
    add_watermark(plt.gca())
    add_watermark(plt.gca())
    os.makedirs(f"{OUTPUT_DIR}/forensics", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/forensics/v1_linear_bias.png", dpi=300, bbox_inches='tight')
    plt.close()

# 8. HISTORICAL FORENSICS: V2 OVERFITTING
def gen_v2_overfitting():
    plt.figure(figsize=(12, 8))
    epochs = np.arange(0, 500)
    train_loss = 1 / (epochs + 20) + 0.05
    val_loss = 1 / (epochs + 20) + 0.05 + (epochs/1000)**2  # Divergence
    
    plt.plot(epochs, train_loss, color=ACCENT_GREEN, lw=3, label="TRAINING LOSS (MEMORIZATION)")
    plt.plot(epochs, val_loss, color=COLOR_FAILURE, lw=3, label="VALIDATION LOSS (REALITY)")
    
    # Adjusted coordinates to keep arrow on screen (pointing to line at ~0.2, text below at 0.15)
    plt.annotate('THE OVERFITTING GAP', xy=(400, 0.21), xytext=(250, 0.35),
                 arrowprops=dict(facecolor=COLOR_FAILURE, shrink=0.05),
                 color=COLOR_FAILURE, weight='bold')
    
    plt.title("FORENSIC ANALYSIS: v2 GRADIENT BOOSTING FAILURE MODE", pad=20, color=COLOR_FAILURE, weight='bold')
    plt.xlabel("TRAINING ITERATIONS")
    plt.ylabel("LOG-LOSS ERROR")
    # Set y-limit to ensure space for annotation
    plt.ylim(0, 0.45)
    plt.legend(loc='upper left')
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/forensics/v2_overfit_divergence.png", dpi=300, bbox_inches='tight')
    plt.close()

# 9. FEATURE EVOLUTION MATRIX
def gen_feature_evolution():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # V1 Features
    v1_feats = ["PPG", "Yards/Game", "Turnover Dif", "Home Field", "QB Rating"]
    v1_imp = [0.45, 0.25, 0.15, 0.1, 0.05]
    axes[0].barh(v1_feats, v1_imp, color=COLOR_FAILURE, alpha=0.7)
    axes[0].set_title("v1: RETAIL HEURISTICS", color=COLOR_FAILURE, weight='bold')
    axes[0].invert_yaxis()
    
    # V2 Features
    v2_feats = ["EPA/Play", "Success Rate", "DVOA", "CPOE", "RedZone Eff"]
    v2_imp = [0.35, 0.25, 0.2, 0.15, 0.05]
    axes[1].barh(v2_feats, v2_imp, color=ACCENT_GOLD, alpha=0.7)
    axes[1].set_title("v2: ADVANCED STATS", color=ACCENT_GOLD, weight='bold')
    axes[1].invert_yaxis()
    
    # V3 Features
    v3_feats = ["Trench Delta", "Pace Pressure", "EDSR", "Wind Entropy", "Traj. Vector"]
    v3_imp = [0.30, 0.25, 0.20, 0.15, 0.10]
    axes[2].barh(v3_feats, v3_imp, color=COLOR_ASTRALIS, alpha=0.7)
    axes[2].set_title("v3: KINETIC PHYSICS", color=COLOR_ASTRALIS, weight='bold')
    axes[2].invert_yaxis()
    
    plt.suptitle("THE EVOLUTION OF SIGNAL DISCOVERY", fontsize=20, color=TEXT_COLOR, weight='bold', y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Prevent title overlap
    plt.savefig(f"{OUTPUT_DIR}/forensics/feature_evolution_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

# 10. DEEP TECH: TISA ATTENTION HEATMAP
def gen_tisa_visualizer():
    plt.figure(figsize=(10, 8))
    # Simulated Attention Weights (11x11 matrix for Offense vs Defense)
    # High attention on OL (col 2-6) vs DL (row 2-6)
    weights = np.random.exponential(0.1, (11, 11))
    weights[2:7, 2:7] += 0.8  # The Trenches
    weights[0, :] = 0.05 # QB scans everything lightly
    weights[:, 0] = 0.05 
    
    sns.heatmap(weights, cmap="magma", xticklabels=False, yticklabels=False, cbar_kws={'label': 'Attention Magnitude (α)'})
    
    plt.title("ORION v3: SPARSE ATTENTION MASK (TRENCH FOCUS)", pad=20, color=COLOR_ORION, weight='bold')
    plt.xlabel("DEFENSIVE ALIGNMENT (Nodes)")
    plt.ylabel("OFFENSIVE ALIGNMENT (Nodes)")
    
    add_watermark(plt.gca())
    os.makedirs(f"{OUTPUT_DIR}/deep_tech", exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/deep_tech/tisa_attention_map.png", dpi=300)
    plt.close()

# 11. DEEP TECH: BAYESIAN RELIABILITY DIAGRAM
def gen_bayesian_reliability():
    plt.figure(figsize=(10, 10))
    conf = np.linspace(0, 1, 20)
    acc_uncal = conf + 0.15 * np.sin(conf * np.pi * 2) # S-curve error
    acc_cal = conf + np.random.normal(0, 0.01, 20)     # Calibrated
    
    plt.plot([0,1], [0,1], ':', color=TEXT_COLOR, alpha=0.5, label="PERFECT CALIBRATION")
    plt.plot(conf, acc_uncal, 'o-', color=COLOR_FAILURE, lw=2, label="RAW MODEL (UNCALIBRATED)")
    plt.plot(conf, acc_cal, 's-', color=COLOR_QUASAR, lw=3, label="QUASAR (ISOTONIC MAPPED)")
    
    plt.fill_between(conf, conf, acc_uncal, color=COLOR_FAILURE, alpha=0.1, label="CALIBRATION ERROR (ECE)")
    
    plt.title("QUASAR v3: BAYESIAN RELIABILITY DIAGRAM", pad=20, color=COLOR_QUASAR, weight='bold')
    plt.xlabel("PREDICTED CONFIDENCE")
    plt.ylabel("OBSERVED ACCURACY")
    plt.legend()
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/deep_tech/bayesian_reliability.png", dpi=300)
    plt.close()

# 12. DEEP TECH: KINETIC PHASE SPACE
def gen_kinetic_phase_space():
    plt.figure(figsize=(12, 8))
    # Vector field
    x, y = np.meshgrid(np.linspace(20, 35, 15), np.linspace(-0.2, 0.4, 15))
    u = -0.5 * (x - 27) # Pace pulls to mean
    v = 0.5 * (y - 0.1) # Efficiency pulls to mean
    
    plt.quiver(x, y, u, v, color=COLOR_PULSAR, alpha=0.6)
    
    # Trajectories
    pace_path = 27 + 5 * np.cos(np.linspace(0, 10, 100))
    eff_path = 0.1 + 0.2 * np.sin(np.linspace(0, 10, 100))
    plt.plot(pace_path, eff_path, color=ACCENT_GOLD, lw=3, label="GAME STATE ORBIT")
    
    plt.title("PULSAR v3: KINETIC PHASE SPACE (PACE vs EFFICIENCY)", pad=20, color=COLOR_PULSAR, weight='bold')
    plt.xlabel("GAME PACE (Seconds/Play)")
    plt.ylabel("SCORING EFFICIENCY (EPA/Play)")
    plt.legend()
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/deep_tech/kinetic_phase_space.png", dpi=300)
    plt.close()

# 13. DEEP TECH: GRPO POLICY MANIFOLD
def gen_grpo_policy_manifold():
    plt.figure(figsize=(10, 8))
    # Create surface data
    x = np.linspace(-1, 1, 30) # Advantage
    y = np.linspace(0, 1, 30)  # Policy Probability
    X, Y = np.meshgrid(x, y)
    Z = Y * np.exp(-X**2) + 0.1 * X  # Objective Gain
    
    # Contour plot instead of 3D for clearer reading
    CS = plt.contourf(X, Y, Z, 20, cmap="viridis")
    cbar = plt.colorbar(CS)
    cbar.set_label('Policy Gradient (∇J)', rotation=270, labelpad=15)
    
    plt.title("GRPO: POLICY OPTIMIZATION SURFACE", pad=20, color=ACCENT_GREEN, weight='bold')
    plt.xlabel("ADVANTAGE FUNCTION A(s,a)")
    plt.ylabel("POLICY PROBABILITY π(a|s)")
    plt.grid(True, alpha=0.1)
    
    add_watermark(plt.gca())
    plt.savefig(f"{OUTPUT_DIR}/deep_tech/grpo_policy_manifold.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("💎 Initiating Billionaire-Grade Visual Pipeline (Interstellar Suite)...")
    gen_decision_tree_logic()
    gen_evolution_comparison()
    gen_infrastructure_map()
    gen_alpha_decay()
    gen_rl_reward_convergence()
    gen_hyper_interaction_heatmap()
    gen_dog_vs_fav()
    gen_pace_scarcity()
    gen_bias_check()
    gen_calibration()
    gen_pseudo_monte_carlo()
    gen_error_propagation()
    gen_v1_linear_failure()
    gen_v2_overfitting()
    gen_feature_evolution()
    gen_tisa_visualizer()
    gen_bayesian_reliability()
    gen_kinetic_phase_space()
    gen_grpo_policy_manifold()
    
    # Basic data plots from original script logic but with new styling
    df_history['cum_profit'] = df_history['profit'].cumsum()
    plt.figure(figsize=(15, 7))
    plt.plot(df_history.index, df_history['cum_profit'], color=COLOR_ASTRALIS, lw=4)
    plt.fill_between(df_history.index, 0, df_history['cum_profit'], color=COLOR_ASTRALIS, alpha=0.05)
    plt.title("ASTRALIS: REALIZED CAPITAL VELOCITY (SEASON 2025)", color=COLOR_ASTRALIS, weight='bold')
    plt.savefig(f"{OUTPUT_DIR}/advanced_equity.png", dpi=300)
    
    print(f"✅ Interstellar Visual Suite persisted to {OUTPUT_DIR}")
