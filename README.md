# 🏈 NFL XGBoost Handicapper

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
