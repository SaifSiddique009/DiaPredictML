import optuna
from src.utils.helpers import cross_validate, logger
from src.models.ann import build_ann_model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scikeras.wrappers import KerasClassifier

def tune_model(model_type, X_train, y_train):
    """
    Optuna Bayesian optimization for hyperparameters.
    
    Args:
        model_type (str): 'RandomForest', 'XGBoost', or 'ANN'
    
    Returns:
        Best params.
    """
    def objective(trial):
        if model_type == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'class_weight': 'balanced'
            }
            model = RandomForestClassifier(**params, random_state=42)
        
        elif model_type == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train)
            }
            model = xgb.XGBClassifier(**params, random_state=42)
        
        elif model_type == 'ANN':
            params = {
                'units': trial.suggest_int('units', 16, 128, step=16),
                'layers': trial.suggest_int('layers', 1, 5),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
            }
            def create_model():
                return build_ann_model(**params, input_shape=(X_train.shape[1],))
            model = KerasClassifier(model=create_model, epochs=50, batch_size=32, verbose=0, class_weight='balanced')
        
        score = cross_validate(model, X_train, y_train)
        return score
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())  # Bayesian-like
    study.optimize(objective, n_trials=20)  # Adjust trials for speed
    logger.info(f"Best params for {model_type}: {study.best_params}")
    return study.best_params