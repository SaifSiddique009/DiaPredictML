# DiaPredictML: Diabetes Risk Prediction using Machine Learning and Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/SaifSiddique009/DiaPredictML/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SaifSiddique009/DiaPredictML/pulls)
<!-- [![Issues](https://img.shields.io/github/issues/yourusername/DiaPredictML.svg)](https://github.com/yourusername/DiaPredictML/issues)
[![Forks](https://img.shields.io/github/forks/yourusername/DiaPredictML.svg)](https://github.com/yourusername/DiaPredictML/network)
[![Stars](https://img.shields.io/github/stars/yourusername/DiaPredictML.svg)](https://github.com/yourusername/DiaPredictML/stargazers) -->

## Project Overview

DiaPredictML is a comprehensive machine learning project designed to predict the onset of diabetes using the Pima Indians Diabetes dataset. This dataset contains medical records from female patients of Pima Indian heritage, including features such as glucose levels, BMI, insulin, and age, with the target being a binary indicator of diabetes diagnosis.

The project follows a structured ML pipeline: data loading, exploratory data analysis (EDA), preprocessing, baseline modeling, advanced ML modeling, deep learning with ANN, cross-validation, and hyperparameter tuning using Optuna. It demonstrates best practices in modular code organization, logging, and evaluation metrics tailored for imbalanced classification problems (e.g., F1-score focus).

This repository serves as a showcase for junior ML engineers, highlighting skills in classical ML, deep learning, optimization, and reproducible workflows. The project emphasizes interpretability, efficiency, and handling real-world data challenges like class imbalance.

## Key Features

- **Exploratory Data Analysis (EDA)**: In-depth analysis of data distributions, correlations, and imbalances to inform model selection. Available as both a Python module (`src/eda/explore.py`) and an interactive Jupyter notebook (`notebooks/eda.ipynb`).
- **Baseline Models**: Simple models like Logistic Regression and Decision Tree for quick performance benchmarks.
- **Advanced ML Models**: Ensemble and boosting techniques including Random Forest and XGBoost, with class weighting for imbalance.
- **Deep Learning**: A tunable Artificial Neural Network (ANN) using TensorFlow/Keras, integrated with dropout and early stopping.
- **Cross-Validation**: 5-fold stratified K-Fold CV to ensure robust performance estimates.
- **Hyperparameter Tuning**: Bayesian optimization via Optuna for efficient tuning of all models.
- **Logging and Reproducibility**: All operations logged to `log/project.log` for traceability.
- **Evaluation Metrics**: Focus on accuracy, precision, recall, and F1-score, suitable for imbalanced datasets.
- **Modular Structure**: Code organized into packages for reusability and maintainability.

## Repository Structure

```
DiaPredictML/
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes dataset (download from Kaggle)
├── log/
│   └── project.log           # Log file for all operations
├── notebooks/
│   └── eda.ipynb             # Interactive Jupyter notebook for EDA
├── results/
│   ├── eda_plots/            # Generated plots from EDA (e.g., correlations, distributions)
│   └── (other generated files: model metrics, saved models)
├── src/
│   ├── __init__.py
│   ├── eda/
│   │   └── explore.py        # EDA module
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ann.py            # ANN model implementation
│   │   ├── adv_models.py     # Advanced ML models (Random Forest, XGBoost)
│   │   └── baseline.py       # Baseline models (Logistic Regression, Decision Tree)
│   ├── tuning/
│   │   └── optuna_tune.py    # Optuna-based hyperparameter tuning
│   ├── utils/
│   │   └── helpers.py        # Utility functions (data loading, evaluation, etc.)
├── main.py                   # Main script to run the full pipeline
├── README.md                 # This file
├── pyproject.toml            # UV-managed dependencies
├── uv.lock                   # UV lock file for reproducibility
├── requirements.txt          # Fallback dependencies for pip
└── .gitignore                # Git ignore file
```

## Prerequisites

- Python 3.8 or higher
- Access to the Pima Indians Diabetes dataset: Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in the `data/` directory.
- Optional: GPU for faster ANN training (TensorFlow will auto-detect if available).

## Installation

This project supports two installation methods: UV (recommended for speed and reproducibility) or pip (standard fallback).

### Using UV (Recommended)
UV is a fast Python package and project manager. Install it via the official instructions: [UV Documentation](https://docs.astral.sh/uv/).

1. Clone the repository:
   ```
   git clone https://github.com/SaifSiddique009/DiaPredictML.git
   cd DiaPredictML
   ```

2. Create and activate a virtual environment:
   ```
   uv venv --python 3.12  # Creates .venv with Python 3.12
   ```

3. Install dependencies:
   ```
   uv sync  # Installs from pyproject.toml and uv.lock
   ```

### Using pip
1. Clone the repository (as above).

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Unix/Mac; on Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run Locally

1. Ensure the dataset is in `data/diabetes.csv`.
2. Run the main pipeline:
   ```
   uv run python main.py  # Or just python main.py if using pip
   ```
   This will perform EDA, train models, tune hyperparameters, evaluate, and save results/logs.

Logs will be saved to `log/project.log`, plots to `results/eda_plots/`, and metrics/models to `results/`.

## Running in Google Colab
You can run this project in Google Colab for easy demonstration (uses pip).

1. Open a new notebook: [Google Colab](https://colab.research.google.com).
2. Clone and set up:
   ```bash
   !git clone https://github.com/SaifSiddique009/DiaPredictML.git
   %cd DiaPredictML
   !pip install -r requirements.txt
   ```
3. Run the main script:
   ```python
   !python main.py
   ```
4. View plots in the Files tab (`/content/DiaPredictML/results/*.png`).


Note: Colab provides free GPU/TPU access, which speeds up ANN training. For EDA, open `notebooks/eda.ipynb` directly in Colab.

## Results

The project evaluates models using 5-fold cross-validation and test set metrics. Below are the final metrics on the test set after tuning:

### Baseline Models
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.734   | 0.603    | 0.704 | 0.650   |
| Decision Tree     | 0.708   | 0.605    | 0.481 | 0.536   |

### Advanced ML Models
| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 0.740   | 0.621    | 0.667 | 0.643   |
| XGBoost      | 0.740   | 0.613    | 0.704 | 0.655   |

### ANN Model
| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 0.747   | 0.609    | 0.778 | 0.683   |

Key Observations:
- ANN slightly outperforms ML models in F1-score, likely due to capturing non-linear patterns.
- All models handle class imbalance via weighting, with recall being higher for positive class (diabetes).
- Cross-validation ensured no overfitting; test metrics align closely with CV averages.

## Future Improvements

- **Feature Engineering**: Add domain-specific features (e.g., interaction terms like Glucose*Insulin) or use techniques like PCA for dimensionality reduction.
- **Imbalance Handling**: Experiment with SMOTE or undersampling for further recall improvement.
- **Model Deployment**: Integrate with Flask/FastAPI for a web API, or Streamlit for an interactive dashboard.
- **Ensemble Stacking**: Combine top models (e.g., XGBoost + ANN) via stacking for hybrid performance.
- **Explainability**: Add SHAP or LIME for feature importance visualizations.
- **Scalability**: Test on larger datasets (e.g., augmented versions) or integrate with cloud services like AWS SageMaker.

For questions or contributions, open an issue or pull request!