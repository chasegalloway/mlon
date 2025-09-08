"""
Visualization utilities for machine learning models and data.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import pandas as pd

class Visualizer:
    """A utility class for visualizing machine learning data and results."""
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 8),
                            cmap: str = 'Blues') -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: The confusion matrix to plot
            class_names: Names of the classes
            figsize: Size of the figure
            cmap: Color map for the plot
        """
        plt.figure(figsize=figsize)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=cmap)
        
        if class_names:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
            
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curve(train_scores: List[float],
                          val_scores: List[float],
                          figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot learning curves.
        
        Args:
            train_scores: List of training scores
            val_scores: List of validation scores
            figsize: Size of the figure
        """
        plt.figure(figsize=figsize)
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'b-', label='Training Score')
        plt.plot(epochs, val_scores, 'r-', label='Validation Score')
        
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance: np.ndarray,
                              feature_names: List[str],
                              figsize: Tuple[int, int] = (12, 6),
                              top_n: Optional[int] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            importance: Array of feature importance scores
            feature_names: Names of the features
            figsize: Size of the figure
            top_n: Number of top features to show
        """
        if top_n:
            idx = np.argsort(importance)[-top_n:]
            importance = importance[idx]
            feature_names = np.array(feature_names)[idx]
        
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(feature_names))
        
        plt.barh(y_pos, importance)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_distribution(data: Union[np.ndarray, pd.Series],
                         title: str = 'Distribution Plot',
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot distribution of a feature.
        
        Args:
            data: Data to plot
            title: Title of the plot
            figsize: Size of the figure
        """
        plt.figure(figsize=figsize)
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame,
                              figsize: Tuple[int, int] = (12, 10),
                              cmap: str = 'coolwarm') -> None:
        """
        Plot correlation matrix.
        
        Args:
            data: DataFrame containing the features
            figsize: Size of the figure
            cmap: Color map for the plot
        """
        correlation = data.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(correlation, annot=True, cmap=cmap, center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
