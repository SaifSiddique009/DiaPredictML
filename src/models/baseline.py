from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.utils.helpers import cross_validate, evaluate_model, logger

def train_baselines(X_train, y_train, X_test, y_test):
    """
    Train and evaluate baseline models with CV.
    """
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    cv_score = cross_validate(lr, X_train, y_train)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['LogisticRegression'] = evaluate_model(y_test, y_pred, 'LogisticRegression')
    logger.info(f"LogisticRegression CV F1: {cv_score}")
    
    # Decision Tree
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    cv_score = cross_validate(dt, X_train, y_train)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results['DecisionTree'] = evaluate_model(y_test, y_pred, 'DecisionTree')
    logger.info(f"DecisionTree CV F1: {cv_score}")
    
    return results