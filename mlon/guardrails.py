"""
Automatic guardrails for machine learning to detect data leakage, bias, and fairness issues.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency, ks_2samp

class LeakageDetector:
    """Detect potential data leakage issues in machine learning pipelines."""
    
    @staticmethod
    def check_train_test_overlap(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               threshold: float = 0.01) -> Dict[str, List[str]]:
        """
        Check for overlap between training and test sets.
        
        Args:
            X_train: Training features
            X_test: Test features
            threshold: Maximum allowed overlap percentage
            
        Returns:
            Dictionary with overlap warnings
        """
        warnings = {
            'exact_duplicates': [],
            'high_overlap_features': []
        }
        
        # Check for exact duplicates
        if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
            train_records = set(X_train.astype(str).apply(tuple, axis=1))
            test_records = set(X_test.astype(str).apply(tuple, axis=1))
            overlap = train_records.intersection(test_records)
            
            if len(overlap) > 0:
                warnings['exact_duplicates'].append(
                    f"Found {len(overlap)} exact duplicate records between train and test sets"
                )
        
        # Check feature-level overlap
        for col in X_train.columns:
            if X_train[col].dtype in [np.number]:
                train_values = set(X_train[col].dropna())
                test_values = set(X_test[col].dropna())
                overlap = len(train_values.intersection(test_values))
                overlap_pct = overlap / len(train_values)
                
                if overlap_pct > threshold:
                    warnings['high_overlap_features'].append(
                        f"Feature '{col}' has {overlap_pct:.1%} overlap between train and test"
                    )
        
        return warnings
    
    @staticmethod
    def detect_target_leakage(X: pd.DataFrame,
                            y: pd.Series,
                            threshold: float = 0.8) -> List[Dict[str, Union[str, float]]]:
        """
        Detect potential target leakage in features.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Correlation/mutual information threshold
            
        Returns:
            List of features with potential target leakage
        """
        leakage_warnings = []
        
        for col in X.columns:
            if X[col].dtype in [np.number] and y.dtype in [np.number]:
                # For numerical features and target, use correlation
                correlation = abs(X[col].corr(y))
                if correlation > threshold:
                    leakage_warnings.append({
                        'feature': col,
                        'type': 'correlation',
                        'value': correlation,
                        'warning': f"Feature '{col}' has {correlation:.2f} correlation with target"
                    })
            else:
                # For categorical features or target, use mutual information
                mi_score = mutual_info_score(X[col], y)
                if mi_score > threshold:
                    leakage_warnings.append({
                        'feature': col,
                        'type': 'mutual_information',
                        'value': mi_score,
                        'warning': f"Feature '{col}' has {mi_score:.2f} mutual information with target"
                    })
        
        return leakage_warnings
    
    @staticmethod
    def detect_future_leakage(timestamps: pd.Series,
                            features: pd.DataFrame) -> List[str]:
        """
        Detect features that might contain future information.
        
        Args:
            timestamps: Series of timestamps
            features: Feature matrix
            
        Returns:
            List of features with potential future leakage
        """
        warnings = []
        
        # Sort data by timestamp
        sorted_idx = timestamps.sort_values().index
        
        for col in features.columns:
            if features[col].dtype in [np.number]:
                # Check if feature values are too correlated with future values
                current_values = features.loc[sorted_idx[:-1], col]
                future_values = features.loc[sorted_idx[1:], col]
                correlation = abs(current_values.corr(future_values))
                
                if correlation > 0.95:  # High autocorrelation might indicate future leakage
                    warnings.append(
                        f"Feature '{col}' shows potential future information leakage "
                        f"(autocorrelation: {correlation:.2f})"
                    )
        
        return warnings


class BiasDetector:
    """Detect bias and fairness issues in machine learning models."""
    
    @staticmethod
    def calculate_disparate_impact(predictions: np.ndarray,
                                 protected_feature: pd.Series,
                                 positive_label: int = 1) -> Dict[str, float]:
        """
        Calculate disparate impact metrics for protected groups.
        
        Args:
            predictions: Model predictions
            protected_feature: Protected attribute values
            positive_label: Label considered as positive outcome
            
        Returns:
            Dictionary with disparate impact metrics
        """
        metrics = {}
        
        # Calculate positive prediction rate for each group
        groups = protected_feature.unique()
        base_group = groups[0]
        base_rate = (predictions[protected_feature == base_group] == positive_label).mean()
        
        for group in groups[1:]:
            group_rate = (predictions[protected_feature == group] == positive_label).mean()
            impact_ratio = group_rate / base_rate if base_rate > 0 else float('inf')
            
            metrics[f'disparate_impact_{group}'] = impact_ratio
            
            # Flag if disparate impact is significant (less than 0.8 or greater than 1.25)
            if impact_ratio < 0.8 or impact_ratio > 1.25:
                metrics[f'disparate_impact_warning_{group}'] = (
                    f"Group '{group}' shows disparate impact ratio of {impact_ratio:.2f} "
                    f"compared to base group '{base_group}'"
                )
        
        return metrics
    
    @staticmethod
    def calculate_group_fairness_metrics(y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       protected_feature: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calculate various group fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_feature: Protected attribute values
            
        Returns:
            Dictionary with group fairness metrics
        """
        metrics = {}
        groups = protected_feature.unique()
        
        for group in groups:
            group_mask = protected_feature == group
            
            # True Positive Rate (Equal Opportunity)
            positive_mask = y_true == 1
            tpr = ((y_pred == 1) & (y_true == 1) & group_mask).sum() / (positive_mask & group_mask).sum()
            
            # False Positive Rate
            negative_mask = y_true == 0
            fpr = ((y_pred == 1) & (y_true == 0) & group_mask).sum() / (negative_mask & group_mask).sum()
            
            # Positive Predictive Value
            ppv = ((y_true == 1) & (y_pred == 1) & group_mask).sum() / ((y_pred == 1) & group_mask).sum()
            
            metrics[str(group)] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'positive_predictive_value': ppv
            }
        
        return metrics
    
    @staticmethod
    def check_dataset_bias(data: pd.DataFrame,
                          protected_features: List[str]) -> Dict[str, List[str]]:
        """
        Check for potential bias in the dataset itself.
        
        Args:
            data: Input DataFrame
            protected_features: List of protected attribute columns
            
        Returns:
            Dictionary with bias warnings
        """
        warnings = {
            'representation_bias': [],
            'correlation_bias': []
        }
        
        for protected_feature in protected_features:
            # Check for representation bias
            value_counts = data[protected_feature].value_counts(normalize=True)
            if (value_counts < 0.1).any():  # Less than 10% representation
                warnings['representation_bias'].append(
                    f"Protected feature '{protected_feature}' has underrepresented groups: "
                    f"{value_counts[value_counts < 0.1].index.tolist()}"
                )
            
            # Check for correlation with other features
            for col in data.columns:
                if col != protected_feature and data[col].dtype in [np.number]:
                    correlation = abs(
                        data.groupby(protected_feature)[col].mean().std() / data[col].std()
                    )
                    if correlation > 0.5:  # High correlation with protected attribute
                        warnings['correlation_bias'].append(
                            f"Feature '{col}' shows high correlation ({correlation:.2f}) "
                            f"with protected attribute '{protected_feature}'"
                        )
        
        return warnings
