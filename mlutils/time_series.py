"""
Time series utilities for machine learning data preparation and analysis.
"""
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

class TimeSeriesUtils:
    """A utility class for time series data processing and feature engineering."""
    
    @staticmethod
    def create_sequences(data: Union[np.ndarray, pd.Series],
                        seq_length: int,
                        target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            seq_length: Length of input sequences
            target_horizon: Number of future time steps to predict
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        data = np.array(data)
        
        for i in range(len(data) - seq_length - target_horizon + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length:i + seq_length + target_horizon])
            
        return np.array(X), np.array(y)
    
    @staticmethod
    def add_time_features(df: pd.DataFrame,
                         datetime_column: str) -> pd.DataFrame:
        """
        Add time-based features to the dataset.
        
        Args:
            df: Input DataFrame
            datetime_column: Name of the datetime column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # Extract basic time features
        df['year'] = df[datetime_column].dt.year
        df['month'] = df[datetime_column].dt.month
        df['day'] = df[datetime_column].dt.day
        df['day_of_week'] = df[datetime_column].dt.dayofweek
        df['hour'] = df[datetime_column].dt.hour
        
        # Add cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        return df
    
    @staticmethod
    def calculate_rolling_features(data: Union[np.ndarray, pd.Series],
                                 windows: List[int] = [3, 7, 30]) -> pd.DataFrame:
        """
        Calculate rolling statistics features.
        
        Args:
            data: Input time series data
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        features = pd.DataFrame()
        
        for window in windows:
            features[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            features[f'rolling_std_{window}'] = data.rolling(window=window).std()
            features[f'rolling_min_{window}'] = data.rolling(window=window).min()
            features[f'rolling_max_{window}'] = data.rolling(window=window).max()
            
        return features
    
    @staticmethod
    def detect_seasonality(data: Union[np.ndarray, pd.Series],
                          max_period: int = 365) -> Optional[int]:
        """
        Detect seasonality in time series data using autocorrelation.
        
        Args:
            data: Input time series data
            max_period: Maximum period to check for seasonality
            
        Returns:
            Detected seasonality period or None if no strong seasonality found
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        data = pd.Series(data).diff().dropna()  # Remove trend
        
        acf = [1.]  # Autocorrelation at lag 0 is always 1
        for lag in range(1, min(len(data) // 2, max_period)):
            acf.append(np.corrcoef(data[lag:], data[:-lag])[0,1])
        
        acf = np.array(acf)
        peaks = np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
        
        if len(peaks) > 0 and np.max(acf[peaks]) > 0.3:  # Threshold for strong seasonality
            return peaks[np.argmax(acf[peaks])]
        return None
