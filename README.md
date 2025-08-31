# MLON
[![PyPI Downloads](https://static.pepy.tech/badge/mlon)](https://pepy.tech/projects/mlon)
A comprehensive utility package for machine learning development that works seamlessly with popular ML libraries like TensorFlow, scikit-learn, Keras, and PyTorch.

## Features

- **Data Preprocessing**
  - Missing value handling
  - Feature scaling
  - Categorical encoding

- **Model Evaluation**
  - Classification metrics
  - Regression metrics
  - Confusion matrix analysis
  - Cross-validation utilities

- **Visualization**
  - Confusion matrix plots
  - Learning curves
  - Feature importance plots
  - Distribution plots
  - Correlation matrices

- **Model Utilities**
  - Model saving/loading
  - Hyperparameter tuning
  - Grid search and random search
  - Model size estimation

- **Cross Validation**
  - K-fold cross-validation
  - Stratified k-fold
  - Custom scoring support

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

## Documentation

For detailed documentation and examples, visit [documentation link].

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
