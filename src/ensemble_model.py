
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

    def get_feature_contributions(self, X):
        """
        Calculates feature contributions (SHAP values) for the input X.
        Returns a list of dictionaries (one per sample), where each dict maps feature -> contribution.
        """
        import pandas as pd
        import xgboost as xgb
        
        final_contribs = np.zeros(X.shape)
        # Note: XGBoost/CatBoost return (n_samples, n_features + 1) where +1 is bias.
        # We need to handle that.
        
        # Initialize with zeros for all samples, all features
        # We'll use a DataFrame to align columns easily
        contrib_df = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        
        for name, model in self.estimators:
            # Determine weight
            if self.weights:
                # Find index of this model
                idx = [n for n, _ in self.estimators].index(name)
                w = self.weights[idx]
                # Normalize weights? self.weights usually sum to 1?
                # If they don't, we should normalize. 
                # Assuming they do or close enough for now.
            else:
                w = 1.0 / len(self.estimators)
                
            try:
                if 'XGB' in str(type(model)).upper():
                    booster = model.get_booster()
                    # feature_names issue: XGBoost might lose them if not careful
                    # But sklearn wrapper usually communicates them.
                    
                    # Create DMatrix
                    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
                    shap_values = booster.predict(dtest, pred_contribs=True)
                    # shap_values is (n_samples, n_features + 1)
                    # The last column is Bias.
                    
                    # Remove bias for now, or just map by index
                    # We assume columns match X.columns in order
                    
                    # Add to totals
                    contrib_df += shap_values[:, :-1] * w
                    
                elif 'CatBoost' in str(type(model)):
                    from catboost import Pool
                    shap_values = model.get_feature_importance(Pool(X), type='ShapValues')
                    # CatBoost SHAP is (n_samples, n_features + 1)
                    contrib_df += shap_values[:, :-1] * w
                    
                else:
                    # Fallback for others (LightGBM or unknown)
                    # Only if they have a simple way. If not, skip.
                    pass
            except Exception as e:
                print(f"Warning: Could not get SHAP for {name}: {e}")
                
        return contrib_df


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

    def get_feature_contributions(self, X):
        """
        Calculates feature contributions (SHAP values) for the input X (Probability space).
        NOTE: Getting SHAP for probabilities is harder. Usually SHAP is for LogOdds in XGB/CatBoost.
        We will approximate or just return LogOdds contributions if possible, 
        but mixing LogOdds from different models is tricky.
        
        SIMPLIFICATION: Just use the primary model or XGBoost if available.
        """
        import pandas as pd
        import xgboost as xgb
        
        contrib_df = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        
        for name, model in self.estimators:
             # Determine weight
            if self.weights:
                idx = [n for n, _ in self.estimators].index(name)
                w = self.weights[idx]
            else:
                w = 1.0 / len(self.estimators)
                
            try:
                if 'XGB' in str(type(model)).upper():
                    booster = model.get_booster()
                    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
                    # pred_contribs returns Margin (LogOdds) contributions
                    shap_values = booster.predict(dtest, pred_contribs=True)
                    contrib_df += shap_values[:, :-1] * w
                    
                elif 'CatBoost' in str(type(model)):
                    from catboost import Pool
                    shap_values = model.get_feature_importance(Pool(X), type='ShapValues')
                    contrib_df += shap_values[:, :-1] * w
            except:
                pass
                
        return contrib_df

