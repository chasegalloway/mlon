"""
Feature selection and dimensionality reduction utilities.
"""
import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FeatureSelector:
    """A utility class for feature selection and dimensionality reduction."""
    
    @staticmethod
    def select_k_best(X: np.ndarray,
                     y: np.ndarray,
                     k: int,
                     method: str = 'f_classif',
                     task: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top k features based on statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Selection method ('f_classif', 'mutual_info', 'f_regress')
            task: Type of task ('classification' or 'regression')
            
        Returns:
            Tuple of (selected features, feature importance scores)
        """
        if task == 'classification':
            if method == 'f_classif':
                selector = SelectKBest(f_classif, k=k)
            else:  # mutual_info
                selector = SelectKBest(mutual_info_classif, k=k)
        else:  # regression
            if method == 'f_regress':
                selector = SelectKBest(f_regression, k=k)
            else:  # mutual_info
                selector = SelectKBest(mutual_info_regression, k=k)
                
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector.scores_
    
    @staticmethod
    def apply_pca(X: np.ndarray,
                  n_components: Optional[Union[int, float]] = None,
                  whiten: bool = False) -> Tuple[np.ndarray, PCA]:
        """
        Apply Principal Component Analysis.
        
        Args:
            X: Feature matrix
            n_components: Number of components or variance ratio to preserve
            whiten: Whether to apply whitening
            
        Returns:
            Tuple of (transformed data, fitted PCA object)
        """
        pca = PCA(n_components=n_components, whiten=whiten)
        X_transformed = pca.fit_transform(X)
        return X_transformed, pca
    
    @staticmethod
    def apply_ica(X: np.ndarray,
                  n_components: int) -> Tuple[np.ndarray, FastICA]:
        """
        Apply Independent Component Analysis.
        
        Args:
            X: Feature matrix
            n_components: Number of independent components
            
        Returns:
            Tuple of (transformed data, fitted ICA object)
        """
        ica = FastICA(n_components=n_components)
        X_transformed = ica.fit_transform(X)
        return X_transformed, ica
    
    @staticmethod
    def apply_lda(X: np.ndarray,
                  y: np.ndarray,
                  n_components: Optional[int] = None) -> Tuple[np.ndarray, LinearDiscriminantAnalysis]:
        """
        Apply Linear Discriminant Analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_components: Number of components
            
        Returns:
            Tuple of (transformed data, fitted LDA object)
        """
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_transformed = lda.fit_transform(X, y)
        return X_transformed, lda
    
    @staticmethod
    def get_feature_importance_mask(importance_scores: np.ndarray,
                                  threshold: float = 0.01) -> np.ndarray:
        """
        Create a boolean mask for important features.
        
        Args:
            importance_scores: Array of feature importance scores
            threshold: Minimum importance score to keep
            
        Returns:
            Boolean mask indicating important features
        """
        return importance_scores > threshold
