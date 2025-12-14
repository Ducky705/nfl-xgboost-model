import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'

DATA_PATH = "data/betting_history.csv"
IMG_DIR = "docs/images"

def generate_visuals():
    if not os.path.exists(DATA_PATH):
        print("❌ No betting history found. Run main.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        print("⚠️ Betting history is empty.")
        return

    # Ensure Directory Exists
    os.makedirs(IMG_DIR, exist_ok=True)

    # --- 1. CUMULATIVE PROFIT (BANKROLL GROWTH) ---
    if 'profit' in df.columns:
        df['cumulative_profit'] = df['profit'].cumsum()
        
        plt.figure()
        sns.lineplot(data=df, x=df.index, y='cumulative_profit', linewidth=2.5, color='#10b981') # Emerald Green
        plt.fill_between(df.index, df['cumulative_profit'], color='#10b981', alpha=0.1)
        plt.title('Bankroll Growth (Cumulative Units Won)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Total Bets Placed', fontsize=12)
        plt.ylabel('Units Won', fontsize=12)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/bankroll_growth.png")
        print(f"✅ Saved {IMG_DIR}/bankroll_growth.png")
        plt.close()

    # --- 2. WIN RATE BY CONFIDENCE ---
    if 'conf' in df.columns:
        # Clean labels (remove emojis/text artifacts)
        df['conf_clean'] = df['conf'].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.strip()
        
        # Filter out "None" or empty
        conf_df = df[df['conf_clean'] != 'None'].copy()
        
        if not conf_df.empty:
            summary = conf_df.groupby('conf_clean')['result'].value_counts(normalize=True).unstack().fillna(0)
            if 'WIN' in summary.columns:
                win_rates = summary['WIN'] * 100
                
                plt.figure(figsize=(8, 5))
                # Custom colors for tiers
                palette = {'STRONG': '#ef4444', 'SOLID': '#22c55e', 'LEAN': '#eab308'}
                
                ax = sns.barplot(x=win_rates.index, y=win_rates.values, palette=palette)
                plt.title('Win Rate by Confidence Level', fontsize=16, fontweight='bold', pad=15)
                plt.ylabel('Win Rate (%)', fontsize=12)
                plt.xlabel('', fontsize=12)
                plt.ylim(0, 100)
                plt.axhline(52.38, color='red', linestyle='--', label='Breakeven (52.4%)')
                plt.legend()
                
                # Add text labels on bars
                for i, v in enumerate(win_rates.values):
                    ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"{IMG_DIR}/win_rate_conf.png")
                print(f"✅ Saved {IMG_DIR}/win_rate_conf.png")
                plt.close()

    # --- 3. ROI DISTRIBUTION ---
    if 'profit' in df.columns:
        plt.figure()
        sns.histplot(df['profit'], bins=15, kde=True, color='#6366f1')
        plt.title('Distribution of Bet Results', fontsize=16, fontweight='bold')
        plt.xlabel('Units Won/Lost per Bet')
        plt.axvline(df['profit'].mean(), color='red', linestyle='dashed', linewidth=1, label=f"Avg: {df['profit'].mean():.2f}u")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{IMG_DIR}/profit_dist.png")
        print(f"✅ Saved {IMG_DIR}/profit_dist.png")
        plt.close()

if __name__ == "__main__":
    print("📊 Generating Visuals...")
    try:
        generate_visuals()
        print("🎉 Charts generated successfully in docs/images/")
    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
