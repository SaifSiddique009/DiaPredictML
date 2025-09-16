from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.eda.explore import perform_eda
from src.utils.helpers import load_and_preprocess_data, evaluate_model, save_results
from src.models.baseline import train_baselines
from src.models.adv_models import train_ml_models
from src.models.ann import train_ann
from src.tuning.optuna_tune import tune_model

def main():
    data_path = 'data/diabetes.csv'
    
    # EDA
    perform_eda(data_path)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    
    # Baselines (no tuning needed)
    baseline_results = train_baselines(X_train, y_train, X_test, y_test)
    save_results(baseline_results, 'baselines')
    
    # Tune and Train ML Models
    ml_results = {}
    for model_type in ['RandomForest', 'XGBoost']:
        best_params = tune_model(model_type, X_train, y_train)
        # Re-train with best params (example for RF; adapt for others)
        if model_type == 'RandomForest':
            model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        elif model_type == 'XGBoost':
            best_params['scale_pos_weight'] = (len(y_train) - sum(y_train)) / sum(y_train)
            model = xgb.XGBClassifier(**best_params, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ml_results[model_type] = evaluate_model(y_test, y_pred, model_type)
    
    save_results(ml_results, 'ml_models')
    
    # Tune and Train ANN
    ann_params = tune_model('ANN', X_train, y_train)
    ann_results, ann_model = train_ann(X_train, y_train, X_test, y_test, params=ann_params)
    save_results(ann_results, 'ann')
    ann_model.save('results/best_ann_model.keras')

if __name__ == '__main__':
    main()