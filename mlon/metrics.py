"""
Model evaluation metrics and utilities.
"""
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report

class ModelEvaluator:
    """A utility class for evaluating machine learning models."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing various classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing various regression metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           normalize: Optional[str] = None) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization strategy ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                target_names: Optional[List[str]] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: List of target class names
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def cross_val_scores(model, X: np.ndarray,
                        y: np.ndarray,
                        cv: int = 5,
                        metric: str = 'accuracy') -> Dict[str, Union[float, List[float]]]:
        """
        Perform cross-validation and return scores.
        
        Args:
            model: The machine learning model
            X: Features
            y: Target variable
            cv: Number of cross-validation folds
            metric: Metric to evaluate ('accuracy', 'precision', 'recall', 'f1', 'mse', 'rmse', 'mae', 'r2')
            
        Returns:
            Dictionary containing cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        return {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std()
        }
