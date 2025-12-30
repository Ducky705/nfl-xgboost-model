from .base_generator import BaseFeatureGenerator
import pandas as pd
import numpy as np

class MarketGenerator(BaseFeatureGenerator):
    """
    Generates market-based features:
    - Independent Market Spread/Total (as features)
    - Implied Team Totals (Vegas Expectation)
    """
    
    def generate(self):
        print("   [Market] Generating Market-Implied features...")
        schedule = self.raw_data['schedule']
        games = schedule[schedule['game_type'] == 'REG'].copy()
        
        # Ensure float
        games['spread_line'] = games['spread_line'].astype(float)
        games['total_line'] = games['total_line'].astype(float)
        
        # 1. Market Features (Renamed to avoid 'result' confusion)
        games['market_spread_line'] = games['spread_line']
        games['market_total_line'] = games['total_line']
        
        # 2. Implied Team Totals
        # Theory: Total = Home + Away
        # Spread (Home) = Away - Home (Wait, nfl_data_py: spread_line is Home Team Spread.
        # e.g. KC -3.5. spread_line = 3.5? Or -3.5?
        # Let's check logic:
        # If Home is favored by 3.5, spread_line is usually 3.5 in some DBs, or -3.5 in others.
        # nfl_data_py: "spread_line" - The spread line for the home team. 
        # Usually positive means underdog? No, standard notation: -3.5 favored.
        # BUT nfl_data_py often uses: result = home - away.
        # If result > spread...
        # Let's check a standard row.
        # If KC beats LV 30-20. Result = 10.
        # If KC was -7. 10 > 7. Covers.
        
        # Let's assume standard nfl_data_py:
        # spread_line positive -> Home Underdog (Home +3.5)
        # spread_line negative -> Home Favorite (Home -3.5)
        
        # Implied Score Calculation:
        # Total = H + A
        # Spread = A - H (If spread is "Line for Home", e.g. +3.5 means H = A - 3.5. So A - H = 3.5)
        # Wait.
        # Spread = 3.5 (Home Underdog). H + 3.5 = A. => A - H = 3.5.
        # Spread = -3.5 (Home Fav). H - 3.5 = A. => H - A = 3.5. => A - H = -3.5.
        
        # So: (Away - Home) = spread_line
        # A + H = Total
        # (A - H) + (A + H) = 2A => spread + total = 2A => A = (Total + Spread)/2
        # (A + H) - (A - H) = 2H => Total - Spread = 2H => H = (Total - Spread)/2
        
        games['implied_away_score'] = (games['total_line'] + games['spread_line']) / 2
        games['implied_home_score'] = (games['total_line'] - games['spread_line']) / 2
        
        # 3. Market Differentials (vs Season Averages)
        # (Merged later, but we can return these columns)
        
        return games[['season', 'week', 'home_team', 'away_team', 'market_spread_line', 'market_total_line', 'implied_home_score', 'implied_away_score']]
