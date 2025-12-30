from .base_generator import BaseFeatureGenerator
import pandas as pd
import numpy as np

class TeamStatsGenerator(BaseFeatureGenerator):
    """
    Generates team-level rolling statistics.
    Ported from original v3 features.py.
    """
    
    def generate(self):
        print("   [TeamStats] Generating Rolling features...")
        base_stats = self.raw_data['base_stats']
        schedule = self.raw_data['schedule']
        
        # Merge betting stats into base stats
        betting_df = self._calculate_betting_stats(schedule)
        full_stats = base_stats.merge(betting_df, on=['season', 'week', 'team'], how='outer')
        
        # Ensure points exist
        if 'points_scored' not in full_stats.columns:
            full_stats = self._add_scores(full_stats, schedule)
            
        # Metrics to roll
        metrics = [
            'off_epa', 'off_ypp', 'off_pass_epa', 'off_rush_epa', 
            'off_edsr', 'off_sack_rate', 'def_sack_rate', 
            'turnovers_lost', 'turnovers_gained', 'off_rz_epa',
            'def_epa', 'def_ypp', 'def_pass_epa', 'def_rush_epa', 'def_edsr',
            'ats_win', 'su_win', 'over_hit',
            'points_scored', 'points_allowed'
        ]
        
        # Generate Rolling Stats
        rolling = self.apply_rolling_stats(full_stats, metrics, windows=[3, 5, 10, 0])
        
        return rolling
        
    def _calculate_betting_stats(self, schedule):
        games = schedule[schedule['game_type'] == 'REG'].copy()
        games['home_margin'] = games['home_score'] - games['away_score']
        games['home_cover'] = (games['home_margin'] > (-1 * games['spread_line'])).astype(int)
        games['away_cover'] = (games['home_margin'] < (-1 * games['spread_line'])).astype(int)
        
        # Push handling
        games.loc[games['home_margin'] == (-1 * games['spread_line']), ['home_cover', 'away_cover']] = 0.5
        
        games['home_su'] = (games['home_score'] > games['away_score']).astype(int)
        games['away_su'] = (games['away_score'] > games['home_score']).astype(int)
        
        games['total_score'] = games['home_score'] + games['away_score']
        games['over_hit'] = (games['total_score'] > games['total_line']).astype(int)
        games.loc[games['total_score'] == games['total_line'], 'over_hit'] = 0.5
        
        # Restructure
        home_df = games[['season', 'week', 'home_team', 'home_cover', 'home_su', 'over_hit']].rename(columns={'home_team': 'team', 'home_cover': 'ats_win', 'home_su': 'su_win'})
        away_df = games[['season', 'week', 'away_team', 'away_cover', 'away_su', 'over_hit']].rename(columns={'away_team': 'team', 'away_cover': 'ats_win', 'away_su': 'su_win'})
        
        return pd.concat([home_df, away_df]).sort_values(['team', 'season', 'week'])

    def _add_scores(self, stats_df, schedule):
        games_tmp = schedule[schedule['game_type'] == 'REG'].copy()
        h = games_tmp[['season', 'week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'points_scored', 'away_score': 'points_allowed'})
        a = games_tmp[['season', 'week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'points_scored', 'home_score': 'points_allowed'})
        scores = pd.concat([h, a])
        return stats_df.merge(scores, on=['season', 'week', 'team'], how='left')
