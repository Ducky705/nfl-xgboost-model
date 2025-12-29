
import pandas as pd
import numpy as np
import pickle

# --- CONFIG ---
CACHE_PATH_V2 = "data/nfl_db_v2.pkl"
FEATURES_PATH_V2 = "data/nfl_features_v2.pkl"
TEAM_MAP = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'SD': 'LAC', 'SL': 'LA', 'STL': 'LA', 'OAK': 'LV'}

import sys

def apply_rolling_stats(stats_df, cols, windows=[3, 5, 0]):
    """
    Applies rolling averages for specified windows.
    Window 0 = Season-to-date expanding mean.
    """
    try:
        df = stats_df.sort_values(['team', 'season', 'week']).copy()
        result = df[['season', 'week', 'team']].copy()
        
        for col in cols:
            # Shift 1 to ensure we only use PAST data for prediction
            shifted = df.groupby(['team', 'season'])[col].shift(1)
            
            for w in windows:
                label = f"{col}_L{w}" if w > 0 else f"{col}_season"
                
                if w == 0:
                    # Expanding mean (Season To Date)
                    # Using transform ensures index alignment
                    result[label] = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.expanding().mean())
                else:
                    # Rolling mean
                    result[label] = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                    
        return result
    except Exception as e:
        print("ERROR inside apply_rolling_stats:")
        import traceback
        traceback.print_exc(file=sys.stdout)
        raise e

def apply_rolling_std(stats_df, cols, windows=[5]):
    """
    Applies rolling standard deviation for specified windows.
    """
    try:
        df = stats_df.sort_values(['team', 'season', 'week']).copy()
        result = df[['season', 'week', 'team']].copy()
        
        for col in cols:
            # Shift 1 to ensure we only use PAST data for prediction
            shifted = df.groupby(['team', 'season'])[col].shift(1)
            
            for w in windows:
                label = f"{col}_std_L{w}"
                # min_periods=3 to avoid high variance on small samples
                result[label] = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.rolling(window=w, min_periods=3).std())
                    
        return result
    except Exception as e:
        print("ERROR inside apply_rolling_std:")
        import traceback
        traceback.print_exc(file=sys.stdout)
        raise e

def calculate_betting_stats(schedule):
    """
    Calculates historic betting trends (ATS, SU, OU) for each team from the schedule.
    Returns a dataframe with columns: season, week, team, ats_win, su_win, over_hit
    """
    print("   Calculating Betting Trends...")
    games = schedule[schedule['game_type'] == 'REG'].copy()
    
    # Calculate Results
    games['home_margin'] = games['home_score'] - games['away_score']
    # nfl_data_py spread_line: Home Team Spread (e.g. -3.5 means Home is favored by 3.5)
    # We want "Did Home Cover?". Home Covers if Margin > -(Spread)
    # e.g. Spread -3.5. Margin must be > +3.5. (e.g. 4). 4 > 3.5.
    games['home_cover'] = (games['home_margin'] > (-1 * games['spread_line'])).astype(int)
    games['away_cover'] = (games['home_margin'] < (-1 * games['spread_line'])).astype(int)
    # Push handling? For now binary (Push = 0 for both is slightly wrong, but simple)
    # Let's do 0.5 for push?
    games.loc[games['home_margin'] == (-1 * games['spread_line']), ['home_cover', 'away_cover']] = 0.5
    
    games['home_su'] = (games['home_score'] > games['away_score']).astype(int)
    games['away_su'] = (games['away_score'] > games['home_score']).astype(int)
    
    games['total_score'] = games['home_score'] + games['away_score']
    games['over_hit'] = (games['total_score'] > games['total_line']).astype(int)
    games.loc[games['total_score'] == games['total_line'], 'over_hit'] = 0.5
    
    # Restructure into Team-Week format
    home_df = games[['season', 'week', 'home_team', 'home_cover', 'home_su', 'over_hit']].rename(columns={'home_team': 'team', 'home_cover': 'ats_win', 'home_su': 'su_win'})
    away_df = games[['season', 'week', 'away_team', 'away_cover', 'away_su', 'over_hit']].rename(columns={'away_team': 'team', 'away_cover': 'ats_win', 'away_su': 'su_win'})
    
    betting_df = pd.concat([home_df, away_df]).sort_values(['team', 'season', 'week'])
    return betting_df

def calculate_sos(schedule, rolling_features):
    """
    Calculates Strength of Schedule (SOS) based on the running average 
    of Opponent's Season-to-Date Efficiency.
    """
    print("   Calculating Strength of Schedule...")
    try:
        # 1. Create Long Schedule
        matchups = pd.concat([
            schedule[['season', 'week', 'home_team', 'away_team']].rename(columns={'home_team': 'team', 'away_team': 'opponent'}),
            schedule[['season', 'week', 'away_team', 'home_team']].rename(columns={'away_team': 'team', 'home_team': 'opponent'})
        ])
        
        # 2. Get Opponent's "Entering" Stats (Season-to-date)
        # rolling_features has [team, season, week, ..._season] stats knowing data BEFORE that week.
        # So if we merge on (opponent, season, week), we get their stats ENTERING the match.
        
        # Relevant metric for strength: off_epa_season, def_epa_season, win_pct (if we had it)
        # We'll use EPA as the gold standard.
        stats_to_use = ['off_epa_season', 'def_epa_season']
        
        # Filter rolling_features for just these columns + keys
        opp_stats = rolling_features[['season', 'week', 'team'] + stats_to_use].rename(columns={'team': 'opponent'})
        
        # Merge
        matchups = matchups.merge(opp_stats, on=['season', 'week', 'opponent'], how='left')
        
        # 3. Calculate Rolling Avg of Opponent Strength for the TEAM
        # We want "Avg Opponent Off EPA" and "Avg Opponent Def EPA" played so far
        matchups = matchups.sort_values(['team', 'season', 'week'])
        
        sos_features = matchups[['season', 'week', 'team']].copy()
        
        # Expanding mean of the opponents' stats
        # Shift(1) is NOT needed here because we want the average of opponents *already played* 
        # BUT for prediction in Week X, we know who we played in Weeks 1-(X-1).
        # However, the row for Week X in 'games' represents the game ABOUT TO BE PLAYED.
        # So we want SOS entering Week X, which is the average of opponents from Weeks 1 to X-1.
        # So yes, we shift.
        
        for col in stats_to_use:
            # We take the expanding mean of the opponents stats column
            # Then shift it down so Week X's value is the average of Weeks 1..X-1
            sos_col = f"sos_{col.replace('_season', '')}" # e.g., sos_off_epa
            
            # Group by team/season, take expanding mean, then shift
            # Actually simpler: Shift first, then expanding mean?
            # No, expanding mean includes current row. 
            # We want Expanding Mean of (Previous Rows).
            # So Shift then Expanding Mean.
            
            check_col = matchups.groupby(['team', 'season'])[col].shift(1)
            sos_features[sos_col] = check_col.groupby([matchups['team'], matchups['season']]).expanding().mean().reset_index(level=[0,1], drop=True)
            
        return sos_features
        
    except Exception as e:
        print(f"⚠️ Error calculating SOS: {e}")
        return pd.DataFrame()

def engineering_pipeline(schedule, base_stats, injury_stats):
    print("Engineering Features (Broad Strategy)...")

    # 0. Integrate Betting Stats into Base Stats
    # We want to treat ats_win, su_win etc just like off_epa for rolling purposes
    # But base_stats might not have all weeks if schedule does? Use Merge.
    betting_df = calculate_betting_stats(schedule)
    betting_df['team'] = betting_df['team'].replace(TEAM_MAP)
    
    # Merge into base_stats. 
    # Warning: base_stats is typically PBP aggregated. Betting stats exist for every game in schedule.
    # We should merge on season/week/team.
    full_stats = base_stats.merge(betting_df, on=['season', 'week', 'team'], how='outer') # Outer to keep all games
    
    # Ensure points_scored/allowed exist in full_stats if they aren't in base_stats
    if 'points_scored' not in full_stats.columns and 'home_score' in schedule.columns:
        # We need to map points from schedule to team-week
        # Actually, calculate_betting_stats does not return scores, but it calculates them.
        # Let's add them to calculate_betting_stats output or map them here.
        # Efficient way:
        games_tmp = schedule[schedule['game_type'] == 'REG'].copy()
        h = games_tmp[['season', 'week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'points_scored', 'away_score': 'points_allowed'})
        a = games_tmp[['season', 'week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'points_scored', 'home_score': 'points_allowed'})
        scores = pd.concat([h, a])
        full_stats = full_stats.merge(scores, on=['season', 'week', 'team'], how='left')
    
    # 1. Broad Feature Generation
    metrics = [
        'off_epa', 'off_ypp', 'off_pass_epa', 'off_rush_epa', 
        'off_edsr', 'off_sack_rate', 'def_sack_rate', 
        'turnovers_lost', 'turnovers_gained', 'off_rz_epa',
        # New Defensive Metrics
        'def_epa', 'def_ypp', 'def_pass_epa', 'def_rush_epa', 'def_edsr',
        # New Betting Metrics
        'ats_win', 'su_win', 'over_hit',
        # Scoring Metrics (Totals)
        'points_scored', 'points_allowed'
    ]
    
    # Generate variations: L3, L5, L10, Season
    print("   Generating Rolling features...")
    rolling_features = apply_rolling_stats(full_stats, metrics, windows=[3, 5, 10, 0])
    
    # Generate Volatility (Std Dev) for Totals
    print("   Generating Volatility features...")
    std_features = apply_rolling_std(full_stats, ['points_scored', 'points_allowed'], windows=[5])
    rolling_features = rolling_features.merge(std_features, on=['season', 'week', 'team'], how='left')

    print(f"   Rolling features shape: {rolling_features.shape}")
    
    # 2. Schedule Setup
    games = schedule[schedule['game_type'] == 'REG'].copy()
    games['home_team'] = games['home_team'].replace(TEAM_MAP)
    games['away_team'] = games['away_team'].replace(TEAM_MAP)
    games['gameday'] = pd.to_datetime(games['gameday'])
    
    # 3. Rest Days (Calculated from Schedule)
    rest_df = pd.concat([
        games[['season', 'week', 'gameday', 'home_team']].rename(columns={'home_team': 'team'}), 
        games[['season', 'week', 'gameday', 'away_team']].rename(columns={'away_team': 'team'})
    ]).sort_values(['team', 'gameday'])
    
    rest_df['rest'] = (rest_df['gameday'] - rest_df.groupby('team')['gameday'].shift(1)).dt.days.fillna(7).clip(upper=14)
    
    # Merge Rest
    games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'home_rest'}).drop(columns=['team'])
    games = games.merge(rest_df[['season', 'week', 'team', 'rest']], left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'team']).rename(columns={'rest': 'away_rest'}).drop(columns=['team'])
    
    # Deduplicate columns BEFORE operations to avoid issues
    games = games.loc[:, ~games.columns.duplicated()]
    
    # --- V2.3: SITUATIONAL FEATURES ---
    print("   Adding Situational Features (v2.3)...")
    
    # 3a. Primetime Games (SNF, MNF, TNF)
    # gametime is in format like '20:20' or '13:00' (Eastern)
    # SNF = Sunday 8pm+, MNF = Monday, TNF = Thursday
    if 'gametime' in games.columns:
        games['gametime_hour'] = pd.to_datetime(games['gametime'], format='%H:%M', errors='coerce').dt.hour.fillna(13)
    else:
        games['gametime_hour'] = 13  # Default 1pm
        
    games['gameday_dow'] = games['gameday'].dt.dayofweek # Monday=0, Sunday=6
    
    # Primetime: Thursday (3), Sunday Night (6 + after 7pm), Monday (0)
    games['is_primetime'] = (
        (games['gameday_dow'] == 3) |  # Thursday Night
        (games['gameday_dow'] == 0) |  # Monday Night
        ((games['gameday_dow'] == 6) & (games['gametime_hour'] >= 19))  # Sunday Night
    ).astype(int)
    
    # 3b. Divisional Games
    # NFL Divisions
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
    
    # Divisional Games (restored for v2.5)
    games['home_div'] = games['home_team'].map(TEAM_TO_DIV)
    games['away_div'] = games['away_team'].map(TEAM_TO_DIV)
    games['is_division_game'] = (games['home_div'] == games['away_div']).astype(int)
    games = games.drop(columns=['home_div', 'away_div'], errors='ignore')
    
    # 3c. Bye Week Effect (Rest > 10 days = Coming off Bye)
    games['home_off_bye'] = (games['home_rest'] >= 10).astype(int)
    games['away_off_bye'] = (games['away_rest'] >= 10).astype(int)
    
    # 3d. Short Week (Rest < 6 days)
    games['home_short_week'] = (games['home_rest'] < 6).astype(int)
    games['away_short_week'] = (games['away_rest'] < 6).astype(int)
    
    # 4. Merge Rolling Stats
    for side in ['home', 'away']:
        games = games.merge(rolling_features, left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
        games = games.rename(columns={c: f'{side}_{c}' for c in rolling_features.columns if c not in ['season', 'week', 'team']})
        games = games.drop(columns=['team'])

    # 5. Merge Injury Stats
    # Note: Injury stats are already "current week" status (who is out TODAY), so no shift needed usually.
    # However, if using historical injury rates, we'd roll. Here we want "Impact of CURRENT missing players".
    if not injury_stats.empty:
        for side in ['home', 'away']:
            games = games.merge(injury_stats, left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
            games = games.rename(columns={
                'starters_out_count': f'{side}_starters_out',
                'qb_starter_out': f'{side}_qb_out'
            })
            games = games.drop(columns=['team', 'position', 'is_likely_out', 'qb_out_count'], errors='ignore')
    
    # Fill NAs (Early season weeks might be missing L3 stats)
    # We fill with Season/Career averages or 0? XGBoost handles NaNs, but 0 is safer for "diffs"
    # Merge SOS
    sos_df = calculate_sos(schedule, rolling_features)
    if not sos_df.empty:
        for side in ['home', 'away']:
            games = games.merge(sos_df, left_on=['season', 'week', f'{side}_team'], right_on=['season', 'week', 'team'], how='left')
            # Rename columns
            games = games.rename(columns={c: f'{side}_{c}' for c in sos_df.columns if c not in ['season', 'week', 'team']})
            games = games.drop(columns=['team'])

    # Deduplicate columns just in case
    games = games.loc[:, ~games.columns.duplicated()]
    
    # Fill NAs
    # We fill with Season/Career averages or 0? XGBoost handles NaNs, but 0 is safer for "diffs"
    print("   Filling NaNs...")
    num_cols = games.select_dtypes(include=[np.number]).columns
    # Exclude betting lines from zero-fill so we can detect missing markets
    cols_to_exclude = ['spread_line', 'total_line', 'result']
    cols_to_fill = [c for c in num_cols if c not in cols_to_exclude]
    games.loc[:, cols_to_fill] = games.loc[:, cols_to_fill].fillna(0)

    # 6. Calculate Differentials (Interaction Terms)
    # The model can learn these, but giving explicit diffs helps convergence
    for m in metrics:
        for w in ['L3', 'L5', 'L10', 'season']:
            base = f"{m}_{w}"
            games[f'diff_{base}'] = games[f'home_{base}'] - games[f'away_{base}']
    
    # Specific Mismatches
    # Ensure columns exist (if injury data was missing)
    for col in ['home_qb_out', 'away_qb_out', 'home_starters_out', 'away_starters_out']:
        if col not in games.columns:
            games[col] = 0

    games['rest_diff'] = games['home_rest'] - games['away_rest']
    games['qb_mismatch'] = games['home_qb_out'] - games['away_qb_out']
    games['injury_diff'] = games['home_starters_out'] - games['away_starters_out']
    
    # 7. Totals Specific Features
    # Pace, Combined EPA
    # 7. Totals Specific Features
    # Pace, Combined EPA
    games['total_pace_L3'] = games['home_off_ypp_L3'] + games['away_off_ypp_L3'] # Proxy for pace/efficiency
    games['total_epa_L3'] = games['home_off_epa_L3'] + games['away_off_epa_L3']
    
    # Scoring Sums (Projected Total)
    games['total_ppg_L3'] = games['home_points_scored_L3'] + games['away_points_scored_L3']
    games['total_ppg_allowed_L3'] = games['home_points_allowed_L3'] + games['away_points_allowed_L3']
    
    # Matchup Summations (Home Offense + Away Defense)
    # High Offense vs Bad Defense = High Score.
    # We want to know aggregate quality.
    games['total_proj_score_L3'] = (games['home_points_scored_L3'] + games['away_points_allowed_L3']) + \
                                   (games['away_points_scored_L3'] + games['home_points_allowed_L3'])
                                   
    games['total_epa_proj'] = (games['home_off_epa_L3'] + games['away_def_epa_L3']) + \
                              (games['away_off_epa_L3'] + games['home_def_epa_L3'])

    # --- Volatility & Over Trends ---
    if 'home_points_scored_std_L5' in games.columns:
        games['total_volatility_L5'] = games['home_points_scored_std_L5'] + games['away_points_scored_std_L5']
    
    if 'home_over_hit_L5' in games.columns:
         games['combined_over_pct_L5'] = (games['home_over_hit_L5'] + games['away_over_hit_L5']) / 2
    
    # --- Weather Scoring Factor ---
    # Temp > 80 is good, Temp < 32 is bad. Wind > 15 is bad.
    # We proxy this into a single 'weather_factor'.
    # Normalize Temp: (Temp - 50) / 10 -> 70 deg = 2.0, 30 deg = -2.0
    # Normalize Wind: (Wind) / 5 -> 15 mph = 3.0
    # Factor = Temp_Score - Wind_Score
    if 'temp' in games.columns and 'wind' in games.columns:
        t = games['temp'].fillna(72)
        w = games['wind'].fillna(0)
        games['weather_scoring_factor'] = ((t - 50) / 10) - (w / 5)
    else:
        games['weather_scoring_factor'] = 0
    
    return games

if __name__ == "__main__":
    print(f"Loading Raw Data: {CACHE_PATH_V2}")
    with open(CACHE_PATH_V2, 'rb') as f:
        data = pickle.load(f)
        
    try:
        full_games = engineering_pipeline(data['schedule'], data['base_stats'], data['injury_stats'])
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)
        exit(1)
    
    print(f"Saving Features: {FEATURES_PATH_V2}")
    print(f"   Shape: {full_games.shape}")
    print(f"   Columns: {len(full_games.columns)}")
    
    with open(FEATURES_PATH_V2, 'wb') as f:
        pickle.dump(full_games, f)
