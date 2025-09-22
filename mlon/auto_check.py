"""
Zero-configuration automated checks for ML pipelines.

This module provides production-grade automated checks for machine learning pipelines,
including data leakage detection, bias detection, and comprehensive reporting.

Example:
    >>> from mlon import AutoChecker
    >>> checker = AutoChecker()
    >>> results = checker.check_data(df)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import seaborn as sns
import io
import json
import hashlib
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .guardrails import LeakageDetector, BiasDetector
from .logging import get_logger
from .exceptions import MLONValueError, MLONRuntimeError

logger = get_logger(__name__)

class AutoChecker:
    """One-line automated checks for ML pipelines."""
    
    def __init__(self):
        self.leakage_detector = LeakageDetector()
        self.bias_detector = BiasDetector()
        
    def infer_data_type(self, data: pd.DataFrame) -> str:
        """Infer if data is time series based on index and columns."""
        if isinstance(data.index, pd.DatetimeIndex):
            return 'time_series'
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            return 'time_series'
        return 'tabular'
    
    def infer_target(self, data: pd.DataFrame) -> Tuple[str, str]:
        """Infer target column and problem type."""
        # Common target column names
        target_names = ['target', 'label', 'y', 'class', 'outcome']
        binary_threshold = 2
        
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in target_names or 'target' in col_lower:
                n_unique = data[col].nunique()
                if n_unique <= binary_threshold:
                    return col, 'classification'
                elif data[col].dtype in [np.number]:
                    return col, 'regression'
        
        # If no obvious target, use the last column
        col = data.columns[-1]
        n_unique = data[col].nunique()
        if n_unique <= binary_threshold:
            return col, 'classification'
        elif data[col].dtype in [np.number]:
            return col, 'regression'
        return col, 'unknown'
    
    def infer_protected_features(self, data: pd.DataFrame) -> List[str]:
        """Infer potential protected attributes."""
        protected_keywords = [
            'gender', 'sex', 'race', 'ethnicity', 'age', 'religion',
            'nationality', 'marital', 'disability', 'education'
        ]
        
        protected_cols = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                protected_cols.append(col)
            # Also check for categorical columns with few unique values
            elif data[col].dtype == 'object' and data[col].nunique() < 10:
                protected_cols.append(col)
        
        return protected_cols
    
    def generate_pdf_report(self, 
                          results: Dict,
                          output_path: str = 'mlon_report.pdf') -> None:
        """Generate a PDF report from check results."""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("MLON AutoCheck Report", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Data Overview
        story.append(Paragraph("Data Overview", styles['Heading2']))
        data_stats = results.get('data_overview', {})
        stats_table = Table([
            ['Metric', 'Value'],
            ['Data Type', data_stats.get('data_type', 'Unknown')],
            ['Problem Type', data_stats.get('problem_type', 'Unknown')],
            ['Number of Samples', data_stats.get('n_samples', 0)],
            ['Number of Features', data_stats.get('n_features', 0)]
        ])
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Leakage Detection
        story.append(Paragraph("Data Leakage Detection", styles['Heading2']))
        leakage_results = results.get('leakage_detection', {})
        for check_type, warnings in leakage_results.items():
            story.append(Paragraph(check_type.replace('_', ' ').title(), styles['Heading3']))
            if warnings:
                for warning in warnings:
                    story.append(Paragraph(f"• {warning}", styles['Normal']))
            else:
                story.append(Paragraph("✓ No issues detected", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Bias Detection
        story.append(Paragraph("Bias Detection", styles['Heading2']))
        bias_results = results.get('bias_detection', {})
        for check_type, warnings in bias_results.items():
            story.append(Paragraph(check_type.replace('_', ' ').title(), styles['Heading3']))
            if warnings:
                for warning in warnings:
                    story.append(Paragraph(f"• {warning}", styles['Normal']))
            else:
                story.append(Paragraph("✓ No issues detected", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Recommendations
        if 'recommendations' in results:
            story.append(Paragraph("Recommendations", styles['Heading2']))
            for rec in results['recommendations']:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        # Build the PDF
        doc.build(story)
    
    def check_data(self, 
                  data: pd.DataFrame,
                  train_data: Optional[pd.DataFrame] = None,
                  test_data: Optional[pd.DataFrame] = None,
                  target_col: Optional[str] = None,
                  protected_features: Optional[List[str]] = None,
                  output_path: str = 'mlon_report.pdf') -> Dict:
        """
        One-line comprehensive check of ML data and model.
        
        Args:
            data: DataFrame containing the data to check
            train_data: Optional training data for split checks
            test_data: Optional test data for split checks
            target_col: Optional target column name
            protected_features: Optional list of protected attribute columns
            output_path: Path to save the PDF report
            
        Returns:
            Dictionary containing all check results
        """
        results = {'recommendations': []}
        
        # Infer data properties
        data_type = self.infer_data_type(data)
        if target_col is None:
            target_col, problem_type = self.infer_target(data)
        else:
            problem_type = 'classification' if data[target_col].nunique() <= 2 else 'regression'
        
        if protected_features is None:
            protected_features = self.infer_protected_features(data)
        
        # Data overview
        results['data_overview'] = {
            'data_type': data_type,
            'problem_type': problem_type,
            'n_samples': len(data),
            'n_features': len(data.columns)
        }
        
        # Leakage detection
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        leakage_results = {}
        
        # Train-test split check
        if train_data is not None and test_data is not None:
            leakage_results['train_test_overlap'] = self.leakage_detector.check_train_test_overlap(
                train_data.drop(columns=[target_col]),
                test_data.drop(columns=[target_col])
            )
        
        # Target leakage check
        leakage_results['target_leakage'] = self.leakage_detector.detect_target_leakage(X, y)
        
        # Time series specific checks
        if data_type == 'time_series':
            if isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index
            else:
                date_col = data.select_dtypes(include=['datetime64']).columns[0]
                timestamps = data[date_col]
            leakage_results['future_leakage'] = self.leakage_detector.detect_future_leakage(
                timestamps,
                X
            )
        
        results['leakage_detection'] = leakage_results
        
        # Bias detection
        if protected_features:
            bias_results = {}
            for feature in protected_features:
                bias_results[f'bias_{feature}'] = self.bias_detector.check_dataset_bias(
                    data,
                    protected_features=[feature]
                )
            results['bias_detection'] = bias_results
        
        # Generate recommendations
        if any(len(warnings) > 0 for warnings in leakage_results.values()):
            results['recommendations'].append(
                "Consider reviewing feature engineering to remove potential data leakage"
            )
        
        if protected_features and any(len(warnings) > 0 for warnings in bias_results.values()):
            results['recommendations'].append(
                "Consider techniques to mitigate bias, such as resampling or fairness constraints"
            )
        
        # Generate PDF report
        self.generate_pdf_report(results, output_path)
        
        return results
