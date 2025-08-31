"""
Preprocessing utilities for machine learning data preparation.
"""
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """A utility class for preprocessing machine learning data."""
    
    def __init__(self):
        self.scalers = {}
        self._fitted = False
    
    def handle_missing_values(self, data: Union[pd.DataFrame, np.ndarray],
                            strategy: str = 'mean',
                            columns: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input data (DataFrame or numpy array)
            strategy: Strategy to handle missing values ('mean', 'median', 'mode', 'zero', 'drop')
            columns: Specific columns to apply the strategy to
            
        Returns:
            Processed data with handled missing values
        """
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns
                
            for col in columns:
                if strategy == 'mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'median':
                    data[col].fillna(data[col].median(), inplace=True)
                elif strategy == 'mode':
                    data[col].fillna(data[col].mode()[0], inplace=True)
                elif strategy == 'zero':
                    data[col].fillna(0, inplace=True)
                elif strategy == 'drop':
                    data.dropna(subset=[col], inplace=True)
                    
        elif isinstance(data, np.ndarray):
            if strategy in ['mean', 'median', 'mode']:
                if strategy == 'mean':
                    fill_value = np.nanmean(data, axis=0)
                elif strategy == 'median':
                    fill_value = np.nanmedian(data, axis=0)
                else:  # mode
                    fill_value = np.nanmean(data, axis=0)  # using mean as approximation for mode
                
                mask = np.isnan(data)
                data[mask] = np.take(fill_value, np.where(mask)[1])
                
            elif strategy == 'zero':
                data = np.nan_to_num(data)
                
            elif strategy == 'drop':
                data = data[~np.isnan(data).any(axis=1)]
                
        return data
    
    def scale_features(self, data: Union[pd.DataFrame, np.ndarray],
                      method: str = 'standard',
                      columns: Optional[List[str]] = None,
                      feature_range: tuple = (0, 1)) -> Union[pd.DataFrame, np.ndarray]:
        """
        Scale features using various scaling methods.
        
        Args:
            data: Input data
            method: Scaling method ('standard' or 'minmax')
            columns: Specific columns to scale
            feature_range: Range for MinMaxScaler
            
        Returns:
            Scaled data
        """
        if not self._fitted:
            self.scalers = {}
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
                
            for col in columns:
                if col not in self.scalers:
                    if method == 'standard':
                        self.scalers[col] = StandardScaler()
                    else:  # minmax
                        self.scalers[col] = MinMaxScaler(feature_range=feature_range)
                        
                data[col] = self.scalers[col].fit_transform(data[[col]])
                
        elif isinstance(data, np.ndarray):
            if not self._fitted:
                if method == 'standard':
                    self.scalers['array'] = StandardScaler()
                else:  # minmax
                    self.scalers['array'] = MinMaxScaler(feature_range=feature_range)
                    
            data = self.scalers['array'].fit_transform(data)
            
        self._fitted = True
        return data
    
    def encode_categorical(self, data: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data: Input DataFrame
            columns: Columns to encode
            method: Encoding method ('onehot' or 'label')
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns
            
        for col in columns:
            if method == 'onehot':
                encoded = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data.drop(col, axis=1), encoded], axis=1)
            elif method == 'label':
                data[col] = pd.Categorical(data[col]).codes
                
        return data
