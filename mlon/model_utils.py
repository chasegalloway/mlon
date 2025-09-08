"""
Utility functions for machine learning model management and operations.
"""
import pickle
from typing import Any, Optional, Dict, List
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

class ModelUtils:
    """A utility class for model operations and management."""
    
    @staticmethod
    def save_model(model: Any,
                  filepath: str,
                  method: str = 'pickle') -> None:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            filepath: Path where to save the model
            method: Saving method ('pickle' or 'joblib')
        """
        if method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:  # joblib
            joblib.dump(model, filepath)
    
    @staticmethod
    def load_model(filepath: str,
                  method: str = 'pickle') -> Any:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            method: Loading method ('pickle' or 'joblib')
            
        Returns:
            Loaded model
        """
        if method == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:  # joblib
            model = joblib.load(filepath)
        return model
    
    @staticmethod
    def grid_search(model: Any,
                   param_grid: Dict,
                   X: np.ndarray,
                   y: np.ndarray,
                   cv: int = 5,
                   n_jobs: int = -1,
                   verbose: int = 1) -> GridSearchCV:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model: The model to tune
            param_grid: Dictionary of parameters to search
            X: Training features
            y: Target variable
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Fitted GridSearchCV object
        """
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        grid_search.fit(X, y)
        return grid_search
    
    @staticmethod
    def random_search(model: Any,
                     param_distributions: Dict,
                     X: np.ndarray,
                     y: np.ndarray,
                     n_iter: int = 10,
                     cv: int = 5,
                     n_jobs: int = -1,
                     verbose: int = 1) -> RandomizedSearchCV:
        """
        Perform randomized search for hyperparameter tuning.
        
        Args:
            model: The model to tune
            param_distributions: Dictionary of parameters to sample from
            X: Training features
            y: Target variable
            n_iter: Number of parameter settings to try
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Fitted RandomizedSearchCV object
        """
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        random_search.fit(X, y)
        return random_search
    
    @staticmethod
    def get_model_size(model: Any) -> str:
        """
        Get the size of a model in memory.
        
        Args:
            model: The model to measure
            
        Returns:
            Size of the model in a human-readable format
        """
        import sys
        bytes_size = sys.getsizeof(pickle.dumps(model))
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} TB"
