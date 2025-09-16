from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.utils.helpers import cross_validate, evaluate_model, logger

def train_ml_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate advanced ML models with CV.
    """
    results = {}
    
    # Random Forest
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    cv_score = cross_validate(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['RandomForest'] = evaluate_model(y_test, y_pred, 'RandomForest')
    logger.info(f"RandomForest CV F1: {cv_score}")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42)
    cv_score = cross_validate(xgb_model, X_train, y_train)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model(y_test, y_pred, 'XGBoost')
    logger.info(f"XGBoost CV F1: {cv_score}")
    
    return results