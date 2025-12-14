import os

# ==========================================
# 1. FILE CONTENTS
# ==========================================

# --- REQUIREMENTS (Added Plotting Libs) ---
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
"""

# --- GENERATE GRAPHS SCRIPT ---
GENERATE_GRAPHS_CONTENT = r'''import pandas as pd
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
'''

# --- README: PROFESSIONAL & TRANSPARENT ---
README_CONTENT = """# 🏈 NFL XGBoost Handicapper

**Automated Institutional-Grade NFL Betting Model**

[![Model Update](https://github.com/Ducky705/nfl-xgboost-model/actions/workflows/update_picks.yml/badge.svg)](https://github.com/Ducky705/nfl-xgboost-model/actions/workflows/update_picks.yml)
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-brightgreen)](https://ducky705.github.io/nfl-xgboost-model/)

## 📊 Performance Visuals
*Live performance charts generated automatically from the model's betting history.*

| **Bankroll Growth** | **Win Rate by Confidence** |
|:---:|:---:|
| ![Bankroll](docs/images/bankroll_growth.png) | ![Win Rates](docs/images/win_rate_conf.png) |

---

## 🧠 Methodology

### 1. The Core Philosophy
This model rejects simple "points scored" metrics in favor of **Efficiency** and **Trench Warfare** analytics. It hypothesizes that a team's consistent ability to move the chains (Success Rate) and disrupt the opponent (Sack/EPA differentials) is more predictive of future spread performance than final scores, which are often noisy.

### 2. Feature Engineering
We utilize `nfl_data_py` to ingest play-by-play data dating back to 2018. For every matchup, we engineer 15+ "Mismatch Metrics", including:

* **EPA/Play Differential:** Expected Points Added per play (Offense vs. Opposing Defense).
* **EDSR (Early Down Success Rate):** How efficiently a team stays ahead of the chains on 1st & 2nd down.
* **Sack Rate Delta:** The difference between a team's offensive sack rate and the opponent's defensive pressure rate.
* **Pythagorean Expectancy:** A "Luck-Adjusted" win probability derived from cumulative points for/against.
* **Home Field Strength:** A dynamic rolling window of how well a specific team performs at home against the spread.

### 3. Model Architecture
* **Algorithm:** **XGBoost Regressor** (Gradient Boosting).
* **Target:** `Home Score - Away Score` (The actual point differential).
* **Validation Strategy:** **Strict Walk-Forward Validation**.
    * *Constraint:* To predict the 2024 season, the model is allowed to train **ONLY** on data from 2018-2023.
    * *Purpose:* This prevents "Data Leakage" (knowing the future) and ensures the ROI shown is realistic.

### 4. Betting Strategy (Kelly Criterion)
We do not use flat staking. Unit sizes are calculated using a modified **Kelly Criterion** to maximize geometric growth while minimizing ruin risk.

* **Formula:** `Kelly % = W - (1-W)/R`
* **Implementation:** We use **5% Fractional Kelly** (0.05 multiplier) to dampen volatility and ensure specific bet sizing tiers.
* **Implied Probability:** Derived from the model's predicted edge against the Vegas spread (using Normal Distribution CDF with a standard deviation of 13.86 points).
* **Caps:** Max bet size is strictly capped at **2.0 units**.

---

## 🚀 Usage

### Automatic
This repository is configured with **GitHub Actions**. It runs automatically:
1.  **Every Tuesday (8:00 UTC):** Retrains the model with the latest week's data.
2.  **Every Day (9:00 UTC):** Checks odds, grades yesterday's bets, updates the dashboard, and regenerates performance graphs.

### Manual
To run it locally on your machine:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Update Database & Train:**
    ```bash
    python update_db.py
    ```
3.  **Generate Picks & Dashboard:**
    ```bash
    python main.py
    ```
4.  **Generate Graphs:**
    ```bash
    python generate_graphs.py
    ```

## ⚠️ Disclaimer
This software is for educational and informational purposes only. It does not constitute financial advice. Sports betting involves significant risk.
"""

# --- GITHUB WORKFLOW: ADDS GRAPH GENERATION ---
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

    - name: Run Graph Generation
      run: python generate_graphs.py

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
    print("🚀 Initializing Visuals & Documentation Update...")
    
    # 1. Requirements
    write_file("requirements.txt", REQUIREMENTS_TXT)
    
    # 2. Graph Script
    write_file("generate_graphs.py", GENERATE_GRAPHS_CONTENT)
    
    # 3. Readme
    write_file("README.md", README_CONTENT)
    
    # 4. Update Workflow to run graphs automatically
    write_file(".github/workflows/update_picks.yml", WORKFLOW_CONTENT)
    
    print("\n🎉 Update Complete!")
    print("-------------------------------------------------------")
    print("Step 1: Run 'pip install -r requirements.txt' (to get plotting libs)")
    print("Step 2: Run 'python generate_graphs.py' to create the initial charts.")
    print("Step 3: Push to GitHub:")
    print("        git add .")
    print("        git commit -m 'Added visuals and methodology'")
    print("        git push")
    print("-------------------------------------------------------")