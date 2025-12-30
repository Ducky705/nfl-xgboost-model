
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np

class VotingEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted Voting Ensemble for Regression.
    Compatible with XGBoost, LightGBM, CatBoost.
    """
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        
    def fit(self, X, y):
        for name, model in self.estimators:
            print(f"   Training {name}...")
            model.fit(X, y)
        return self
        
    def predict(self, X):
        preds = []
        for name, model in self.estimators:
            p = model.predict(X)
            preds.append(p)
        return np.average(preds, axis=0, weights=self.weights)


class VotingClassifierEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted Voting Ensemble for Classification (Probability Averaging).
    """
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for name, model in self.estimators:
            print(f"   Training {name}...")
            model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        probas = []
        for name, model in self.estimators:
            p = model.predict_proba(X)
            probas.append(p)
        return np.average(probas, axis=0, weights=self.weights)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
