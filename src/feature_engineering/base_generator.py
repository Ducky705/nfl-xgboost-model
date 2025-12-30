from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureGenerator(ABC):
    """
    Abstract base class for all feature generators.
    Enforces a consistent interface for the engineering pipeline.
    """
    
    def __init__(self, raw_data):
        """
        Initialize with the raw data dictionary.
        
        Args:
            raw_data (dict): Dictionary containing 'schedule', 'base_stats', 'injury_stats', etc.
        """
        self.raw_data = raw_data
        self.output_df = pd.DataFrame()

    @abstractmethod
    def generate(self):
        """
        Execute the feature generation logic.
        Must return the dataframe with new features.
        """
        pass
    
    def apply_rolling_stats(self, stats_df, cols, windows=[3, 5, 0], group_col='team'):
        """
        Helper: Applies rolling averages for specified windows.
        Window 0 = Season-to-date expanding mean.
        """
        try:
            df = stats_df.sort_values(['team', 'season', 'week']).copy()
            result = df[['season', 'week', 'team']].copy()
            
            for col in cols:
                shifted = df.groupby(['team', 'season'])[col].shift(1)
                
                for w in windows:
                    label = f"{col}_L{w}" if w > 0 else f"{col}_season"
                    
                    if w == 0:
                        # Expanding mean (Season To Date)
                        result[label] = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.expanding().mean())
                    else:
                        # Rolling mean
                        result[label] = shifted.groupby([df['team'], df['season']]).transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                        
            return result
        except Exception as e:
            print(f"⚠️ Error in apply_rolling_stats: {e}")
            raise e
