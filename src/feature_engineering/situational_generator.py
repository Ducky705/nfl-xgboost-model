from .base_generator import BaseFeatureGenerator
import pandas as pd
import numpy as np

class SituationalGenerator(BaseFeatureGenerator):
    """
    Generates situational features: Rest, Travel, Primetime, Divisional info.
    """
    
    def generate(self):
        print("   [Situational] Generating Situational features...")
        schedule = self.raw_data['schedule']
        
        
        games = schedule[schedule['game_type'] == 'REG'].copy()
        
        # Drop pre-existing rest columns if they exist (nfl_data_py sometimes has them)
        games = games.drop(columns=['home_rest', 'away_rest'], errors='ignore')
        
        # Date & Time Handling
        if 'gametime' in games.columns:
            games['gametime_hour'] = pd.to_datetime(games['gametime'], format='%H:%M', errors='coerce').dt.hour.fillna(13)
        else:
            games['gametime_hour'] = 13
            
        games['gameday'] = pd.to_datetime(games['gameday'])
        games['gameday_dow'] = games['gameday'].dt.dayofweek
        
        # 1. Primetime
        games['is_primetime'] = (
            (games['gameday_dow'] == 3) |  # Thursday
            (games['gameday_dow'] == 0) |  # Monday
            ((games['gameday_dow'] == 6) & (games['gametime_hour'] >= 19)) # SNF
        ).astype(int)
        
        # 2. Rest Days
        # We need a unified stack of games to calculate diffs
        rest_df = pd.concat([
            games[['season', 'week', 'gameday', 'home_team']].rename(columns={'home_team': 'team'}),
            games[['season', 'week', 'gameday', 'away_team']].rename(columns={'away_team': 'team'})
        ]).sort_values(['team', 'gameday'])
        
        rest_df['rest'] = (rest_df['gameday'] - rest_df.groupby('team')['gameday'].shift(1)).dt.days.fillna(7).clip(upper=14)
        
        # Merge Rest back
        games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'home_rest'}).drop(columns=['team'])
        games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'away_rest'}).drop(columns=['team'])
        
        # 3. Bye Week & Short Week
        games['home_off_bye'] = (games['home_rest'] >= 10).astype(int)
        games['away_off_bye'] = (games['away_rest'] >= 10).astype(int)
        games['home_short_week'] = (games['home_rest'] < 6).astype(int)
        games['away_short_week'] = (games['away_rest'] < 6).astype(int)
        games['rest_diff'] = games['home_rest'] - games['away_rest']
        
        # 4. Divisional Games
        DIVISIONS = {
            'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_WEST': ['ARI', 'LA', 'SEA', 'SF']
        }
        TEAM_TO_DIV = {team: div for div, teams in DIVISIONS.items() for team in teams}
        
        games['home_div'] = games['home_team'].replace(self.raw_data.get('TEAM_MAP', {})).map(TEAM_TO_DIV)
        games['away_div'] = games['away_team'].replace(self.raw_data.get('TEAM_MAP', {})).map(TEAM_TO_DIV)
        games['is_division_game'] = (games['home_div'] == games['away_div']).astype(int)
        
        return games[['season', 'week', 'home_team', 'away_team', 'game_id', 'is_primetime', 'home_rest', 'away_rest', 'rest_diff', 'home_off_bye', 'away_off_bye', 'is_division_game']]
