# MLON (Machine Learning Operations Network)
[![PyPI Downloads](https://static.pepy.tech/badge/mlon)](https://pepy.tech/projects/mlon)

A comprehensive utility package for machine learning development that works seamlessly with popular ML libraries like TensorFlow, scikit-learn, Keras, and PyTorch. MLON provides an interconnected network of operations for streamlined machine learning workflows.

## Features Overview

### 1. Data Preprocessing (`DataPreprocessor`)
```python
from mlon import DataPreprocessor

preprocessor = DataPreprocessor()
```
- **Missing Value Handling**
  ```python
  # Handle missing values with different strategies
  data = preprocessor.handle_missing_values(data, strategy='mean')  # Options: 'mean', 'median', 'mode', 'zero', 'drop'
  ```
- **Feature Scaling**
  ```python
  # Scale features using StandardScaler or MinMaxScaler
  scaled_data = preprocessor.scale_features(data, method='standard')  # Options: 'standard', 'minmax'
  ```
- **Categorical Encoding**
  ```python
  # Encode categorical variables
  encoded_data = preprocessor.encode_categorical(data, method='onehot')  # Options: 'onehot', 'label'
  ```

### 2. Model Evaluation (`ModelEvaluator`)
```python
from mlon import ModelEvaluator

evaluator = ModelEvaluator()
```
- **Classification Metrics**
  ```python
  # Get comprehensive classification metrics
  metrics = evaluator.classification_metrics(y_true, y_pred)  # Returns accuracy, precision, recall, F1
  ```
- **Regression Metrics**
  ```python
  # Get regression performance metrics
  metrics = evaluator.regression_metrics(y_true, y_pred)  # Returns MSE, RMSE, MAE, R²
  ```
- **Confusion Matrix**
  ```python
  conf_matrix = evaluator.get_confusion_matrix(y_true, y_pred, normalize='true')
  report = evaluator.get_classification_report(y_true, y_pred)
  ```

### 3. Visualization (`Visualizer`)
```python
from mlon import Visualizer

viz = Visualizer()
```
- **Model Performance Visualization**
  ```python
  # Plot confusion matrix
  viz.plot_confusion_matrix(conf_matrix, class_names=classes)
  
  # Plot learning curves
  viz.plot_learning_curve(train_scores, val_scores)
  
  # Plot feature importance
  viz.plot_feature_importance(importance_scores, feature_names)
  ```
- **Data Analysis Visualization**
  ```python
  # Plot distribution of features
  viz.plot_distribution(data['feature'])
  
  # Plot correlation matrix
  viz.plot_correlation_matrix(data)
  ```

### 4. Model Utilities (`ModelUtils`)
```python
from mlon import ModelUtils

model_utils = ModelUtils()
```
- **Model Persistence**
  ```python
  # Save and load models
  model_utils.save_model(model, 'model.pkl', method='pickle')  # Options: 'pickle', 'joblib'
  model = model_utils.load_model('model.pkl', method='pickle')
  ```
- **Hyperparameter Tuning**
  ```python
  # Perform grid search
  best_model = model_utils.grid_search(model, param_grid, X, y)
  
  # Perform random search
  best_model = model_utils.random_search(model, param_dist, X, y)
  ```

### 5. Cross Validation (`CrossValidator`)
```python
from mlon import CrossValidator

cv = CrossValidator(n_splits=5)
```
- **Cross-Validation Operations**
  ```python
  # Perform cross-validation with custom scoring
  scores = cv.cross_validate(model, X, y)
  
  # Get fold indices for custom cross-validation
  train_idx, val_idx = cv.get_fold_indices(X, y)
  ```

### 6. Time Series Utilities (`TimeSeriesUtils`) - NEW!
```python
from mlon import TimeSeriesUtils

ts_utils = TimeSeriesUtils()
```
- **Sequence Creation**
  ```python
  # Create sequences for time series prediction
  X_seq, y_seq = ts_utils.create_sequences(data, seq_length=30, target_horizon=7)
  ```
- **Time Feature Engineering**
  ```python
  # Add time-based features
  df_with_features = ts_utils.add_time_features(df, 'date_column')
  
  # Calculate rolling statistics
  rolling_features = ts_utils.calculate_rolling_features(data, windows=[7, 30, 90])
  
  # Detect seasonality
  seasonality_period = ts_utils.detect_seasonality(data)
  ```

### 7. Feature Selection (`FeatureSelector`) - NEW!
```python
from mlon import FeatureSelector

selector = FeatureSelector()
```
- **Statistical Feature Selection**
  ```python
  # Select top k features
  X_selected, scores = selector.select_k_best(X, y, k=10, method='f_classif')
  ```
- **Dimensionality Reduction**
  ```python
  # Apply PCA
  X_pca, pca = selector.apply_pca(X, n_components=0.95)
  
  # Apply ICA
  X_ica, ica = selector.apply_ica(X, n_components=5)
  
  # Apply LDA
  X_lda, lda = selector.apply_lda(X, y, n_components=2)
  ```

## Installation

```bash
pip install mlon
```

## Quick Start

```python
from mlon import DataPreprocessor, ModelEvaluator, Visualizer, ModelUtils, CrossValidator

# Data Preprocessing
preprocessor = DataPreprocessor()
scaled_data = preprocessor.scale_features(data, method='standard')
encoded_data = preprocessor.encode_categorical(data, method='onehot')

# Model Evaluation
evaluator = ModelEvaluator()
metrics = evaluator.classification_metrics(y_true, y_pred)
conf_matrix = evaluator.get_confusion_matrix(y_true, y_pred)

# Visualization
viz = Visualizer()
viz.plot_confusion_matrix(conf_matrix)
viz.plot_learning_curve(train_scores, val_scores)

# Model Management
model_utils = ModelUtils()
model_utils.save_model(model, 'model.pkl')
best_model = model_utils.grid_search(model, param_grid, X, y)

# Cross Validation
cv = CrossValidator(n_splits=5)
scores = cv.cross_validate(model, X, y)
```

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- Pandas >= 1.1.0
- scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- Joblib >= 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- Issue Tracker: [GitHub Issues](https://github.com/chasegalloway/mlon/issues)
- Documentation: [GitHub README](https://github.com/chasegalloway/mlon#readme)
- Source Code: [GitHub Repository](https://github.com/chasegalloway/mlon)
