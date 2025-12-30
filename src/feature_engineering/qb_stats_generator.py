from .base_generator import BaseFeatureGenerator
import pandas as pd
import numpy as np

class QBStatsGenerator(BaseFeatureGenerator):
    """
    Generates QB-specific rolling statistics.
    Uses 'qb_starters' table to identify who played, and tracks their individual history.
    """
    
    def generate(self):
        print("   [QBStats] Generating QB-specific features...")
        qb_starters = self.raw_data.get('qb_starters')
        pbp = self.raw_data.get('pbp_clean')
        
        if qb_starters is None or qb_starters.empty:
            print("No QB Starters data found. Skipping QB features.")
            return pd.DataFrame()

        # 1. Calculate QB EPA per game (from PBP)
        # We need the QB's actual performance in that specific game
        # Filter pbp for pass plays by that QB
        qb_performances = pbp.groupby(['season', 'week', 'posteam', 'passer_id']).agg({
            'epa': 'mean',
            'play_type': 'count' # Dropbacks
        }).reset_index().rename(columns={'posteam': 'team', 'epa': 'qb_epa_game', 'play_type': 'qb_dropbacks'})
        
        # Merge with Starters to filter only the starter's performance
        # We only care about the starter's stats for the model history
        starter_stats = qb_starters.merge(qb_performances, left_on=['season', 'week', 'team', 'qb_id'], right_on=['season', 'week', 'team', 'passer_id'], how='left')
        
        # 2. Apply Rolling Stats per QB ID
        # Instead of GroupBy Team, we GroupBy QB_ID
        # This allows stats to follow the player (e.g. Cousins MIN -> ATL)
        
        metrics = ['qb_epa_game']
        windows = [3, 10, 0] # L3 (Recent Form), L10 (Season Form), Career
        
        # Use BaseGenerator helper but apply to 'qb_id' grouping?
        # The helper assumes 'team' column. Let's override or adapt.
        # Actually, let's just implement custom rolling here for QB ID
        
        df = starter_stats.sort_values(['qb_id', 'season', 'week']).copy()
        result = df[['season', 'week', 'team', 'qb_id']].copy()
        
        for col in metrics:
            shifted = df.groupby('qb_id')[col].shift(1) # Shift 1 to use past data
            
            for w in windows:
                label = f"qb_epa_L{w}" if w > 0 else "qb_epa_career"
                
                if w == 0:
                    result[label] = shifted.groupby(df['qb_id']).transform(lambda x: x.expanding().mean())
                else:
                    result[label] = shifted.groupby(df['qb_id']).transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        
        # Fill NaNs for Rookies / No History
        # If a QB has no history, we fill with:
        # 1. A slightly negative EPA (replacement level) -> -0.05
        # 2. Or 0.0 (league average)
        # Let's use Replacement Level (-0.1) to penalize unknown QBs
        # Actually, let's use 0.0 for now, as XGBoost handles it.
        result = result.fillna(-0.05)
        
        return result[['season', 'week', 'team', 'qb_epa_L3', 'qb_epa_L10', 'qb_epa_career']]
