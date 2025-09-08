"""
Cross-validation utilities for machine learning models.
"""
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import make_scorer

class CrossValidator:
    """A utility class for cross-validation operations."""
    
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize CrossValidator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle the data
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = None
        self.stratified = False
    
    def setup_kfold(self, stratified: bool = False) -> None:
        """
        Set up K-Fold cross validator.
        
        Args:
            stratified: Whether to use stratified k-fold
        """
        self.stratified = stratified
        if stratified:
            self.kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            self.kfold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
    
    def cross_validate(self,
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      custom_scorer: Optional[Dict] = None) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            model: The model to validate
            X: Features
            y: Target variable
            custom_scorer: Dictionary of custom scoring functions
            
        Returns:
            Dictionary containing scores for each fold
        """
        if self.kfold is None:
            self.setup_kfold(stratified=len(np.unique(y)) > 1)
        
        scores = {
            'train_scores': [],
            'val_scores': []
        }
        
        if custom_scorer:
            scores.update({f'{name}_scores': [] for name in custom_scorer.keys()})
        
        for train_idx, val_idx in self.kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            scores['train_scores'].append(train_score)
            scores['val_scores'].append(val_score)
            
            if custom_scorer:
                for name, scorer in custom_scorer.items():
                    if callable(scorer):
                        score = scorer(model, X_val, y_val)
                        scores[f'{name}_scores'].append(score)
        
        return scores
    
    def get_fold_indices(self, X: np.ndarray,
                        y: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get indices for each fold.
        
        Args:
            X: Features
            y: Target variable (required for stratified k-fold)
            
        Returns:
            List of tuples containing train and validation indices for each fold
        """
        if self.kfold is None:
            self.setup_kfold(stratified=y is not None and len(np.unique(y)) > 1)
        
        if self.stratified and y is None:
            raise ValueError("Target variable y is required for stratified k-fold")
        
        return list(self.kfold.split(X, y))
