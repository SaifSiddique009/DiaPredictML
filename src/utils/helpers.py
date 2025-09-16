import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Load dataset, split into features/target, scale features.
    
    Args:
        file_path (str): Path to CSV file.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def evaluate_model(y_true, y_pred, model_name):
    """
    Compute and log evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    logger.info(f"{model_name} Metrics: {metrics}")
    return metrics

def cross_validate(model, X, y, n_splits=5):
    """
    Perform 5-fold stratified CV.
    
    Returns:
        Average F1 score across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred))
    avg_score = np.mean(scores)
    logger.info(f"CV Average F1: {avg_score}")
    return avg_score

def save_results(metrics, model_name, path='results/'):
    """
    Save metrics to file.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{model_name}_metrics.txt"), 'w') as f:
        f.write(str(metrics))