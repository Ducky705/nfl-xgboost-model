"""
Protocol 705 - Feature Importance Analysis
Uses Permutation Importance to identify weak features for pruning.
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- CONFIG ---
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
MODEL_SPREAD_PATH = "models/v2_spread.json"

def analyze_features():
    print("Loading Data & Model...")
    with open(FEATURES_PATH_V2, 'rb') as f:
        df = pickle.load(f)
    
    spread_model = xgb.Booster()
    spread_model.load_model(MODEL_SPREAD_PATH)
    
    # Prepare Data
    df['result'] = df['home_score'] - df['away_score']
    
    # Evaluation Set (Last Season)
    last_season = df['season'].max()
    eval_df = df[df['season'] == last_season].dropna(subset=['result']).copy()
    
    # Get Model Features
    model_features = spread_model.feature_names
    print(f"Analyzing {len(model_features)} features...")
    
    # Ensure all columns exist
    for c in model_features:
        if c not in eval_df.columns:
            eval_df[c] = 0
    
    X = eval_df[model_features]
    y = eval_df['result']
    
    # --- XGBoost Native Feature Importance (Gain) ---
    print("\n1. XGBoost Gain-Based Importance...")
    importance_dict = spread_model.get_score(importance_type='gain')
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {'feature': k, 'gain': v} for k, v in importance_dict.items()
    ]).sort_values('gain', ascending=False)
    
    # Features NOT in the importance dict have zero gain (never used in splits)
    used_features = set(importance_dict.keys())
    unused_features = [f for f in model_features if f not in used_features]
    print(f"   Features with ZERO gain (never used): {len(unused_features)}")
    
    # Add unused features to importance_df
    unused_df = pd.DataFrame([{'feature': f, 'gain': 0.0} for f in unused_features])
    importance_df = pd.concat([importance_df, unused_df], ignore_index=True)
    
    # --- Classify Features ---
    importance_df['rank'] = importance_df['gain'].rank(ascending=False)
    importance_df['tier'] = pd.cut(
        importance_df['gain'], 
        bins=[-1, 0, 100, 500, 1000, float('inf')],
        labels=['UNUSED', 'WEAK', 'MODERATE', 'STRONG', 'CRITICAL']
    )
    
    # --- Report ---
    print("\n===== FEATURE IMPORTANCE TIERS =====")
    tier_counts = importance_df['tier'].value_counts()
    for tier in ['CRITICAL', 'STRONG', 'MODERATE', 'WEAK', 'UNUSED']:
        count = tier_counts.get(tier, 0)
        print(f"   {tier}: {count} features")
    
    print("\n===== TOP 20 FEATURES =====")
    print(importance_df.head(20).to_string(index=False))
    
    print("\n===== BOTTOM 20 FEATURES (Candidates for Pruning) =====")
    bottom_20 = importance_df.tail(20)
    print(bottom_20.to_string(index=False))
    
    # --- Save Results ---
    importance_df.to_csv("data/feature_importance.csv", index=False)
    print("\n✅ Full analysis saved to data/feature_importance.csv")
    
    # --- Pruning Recommendation ---
    prune_threshold = 50  # Features with gain < 50
    to_prune = importance_df[importance_df['gain'] < prune_threshold]
    print(f"\n===== PRUNING RECOMMENDATION =====")
    print(f"Features below threshold ({prune_threshold} gain): {len(to_prune)}")
    print(f"Recommended to KEEP: {len(importance_df) - len(to_prune)} features")
    
    # Save pruned feature list
    keep_features = importance_df[importance_df['gain'] >= prune_threshold]['feature'].tolist()
    with open("data/pruned_features.txt", 'w') as f:
        f.write('\n'.join(keep_features))
    print(f"✅ Pruned feature list saved to data/pruned_features.txt ({len(keep_features)} features)")
    
    # --- Visualization ---
    plt.figure(figsize=(12, 8))
    top_30 = importance_df.head(30)
    colors = ['#CCFF00' if g > 1000 else '#888888' if g > 500 else '#444444' for g in top_30['gain']]
    plt.barh(top_30['feature'], top_30['gain'], color=colors)
    plt.xlabel('Gain (Importance)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 30 Features by Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('docs/feature_importance.png', dpi=150, facecolor='#0A0A0A', edgecolor='none')
    print("✅ Chart saved to docs/feature_importance.png")
    
    return importance_df

if __name__ == "__main__":
    analyze_features()
