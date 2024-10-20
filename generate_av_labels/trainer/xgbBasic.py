"""
Wrapper for base classifier for VAX pipeline
"""
import numpy as np
# mmwave for noise reduction
# import mmwave.dsp as dsp
# import mmwave.clustering as clu

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
import xgboost
from sklearn.semi_supervised import *
from sklearn.model_selection import *
from sklearn.cluster import *


def privacy_fit(X_train, y_train, X_test=None, return_model=False):
    # filter undetected values
    detected_idx = np.where(~(y_train == 'Undetected'))[0]
    X_train, y_train = X_train[detected_idx], y_train[detected_idx]

    # fit classifier
    base_clf_ = xgboost.XGBClassifier()
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y_train)
    base_clf_.fit(X_train, y_encoded)

    if X_test is not None:
        y_pred_encoded = base_clf_.predict(X_test)
        y_pred_proba = base_clf_.predict_proba(X_test).max(axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred, y_pred_proba = None, None
    if return_model:
        return y_pred, y_pred_proba, base_clf_, label_encoder

    return y_pred, y_pred_proba

def get_clf_():
    base_clf_ = xgboost.XGBClassifier()
    return base_clf_
