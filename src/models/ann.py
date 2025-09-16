from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.utils.helpers import cross_validate, evaluate_model, logger
import numpy as np

def build_ann_model(units=32, layers=1, dropout=0.2, optimizer='adam', input_shape=(8,)):
    """
    Build a simple ANN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for _ in range(layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_ann(X_train, y_train, X_test, y_test, params=None):
    """
    Train ANN with optional tuned params and CV (approximate via sklearn wrapper).
    """
    from scikeras.wrappers import KerasClassifier  # For CV compatibility
    
    if params is None:
        params = {'units': 32, 'layers': 1, 'dropout': 0.2, 'optimizer': 'adam'}
    
    def create_model():
        return build_ann_model(**params)
    
    clf = KerasClassifier(model=create_model, epochs=100, batch_size=32, verbose=1, class_weight='balanced')
    cv_score = cross_validate(clf, X_train, y_train)
    
    model = build_ann_model(**params)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, class_weight={0:1, 1: (len(y_train)-sum(y_train))/sum(y_train)})
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    results = evaluate_model(y_test, y_pred, 'ANN')
    logger.info(f"ANN CV F1: {cv_score}")
    
    return results, model