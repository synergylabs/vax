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
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.semi_supervised import *
from sklearn.model_selection import *
from sklearn.cluster import *
from cleanlab.classification import CleanLearning
from imblearn.over_sampling import SMOTE


def privacy_fit(X_train, y_train, X_test=None, return_model=False):
    # filter undetected values
    detected_idx = np.where(~(y_train == 'Undetected'))[0]
    X_train, y_train = X_train[detected_idx], y_train[detected_idx]

    # list of classifiers for voting
    clfs = [
        SVC(probability=True),
        SVC(C=100,probability=True),
        KNeighborsClassifier(n_neighbors=20),
        KNeighborsClassifier(n_neighbors=40),
        KNeighborsClassifier(n_neighbors=20,weights='distance'),
        KNeighborsClassifier(n_neighbors=40,weights='distance'),
    ]

    # build classifier with clean learning
    base_clf_ = VotingClassifier(clfs,voting='soft')
    clean_clf = CleanLearning(clf=base_clf_)
    label_encoder = LabelEncoder()

    # Balance class distribution using SMOTE
    y_encoded = label_encoder.fit_transform(y_train)
    _, counts = np.unique(y_encoded, return_counts=True)
    smote_model = SMOTE(sampling_strategy='not majority', k_neighbors=min(counts) - 1, n_jobs=6)
    X_sm, y_sm = smote_model.fit_resample(X_train, y_encoded)

    clean_clf.fit(X_sm, y_sm)

    if X_test is not None:
        y_pred_encoded = clean_clf.predict(X_test)
        y_pred_proba = clean_clf.predict_proba(X_test).max(axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred, y_pred_proba = None, None
    if return_model:
        return y_pred, y_pred_proba, base_clf_, label_encoder

    return y_pred, y_pred_proba
def get_clf_():
    # list of classifiers for voting
    clfs = [
        SVC(probability=True),
        SVC(C=100, probability=True),
        KNeighborsClassifier(n_neighbors=20),
        KNeighborsClassifier(n_neighbors=40),
        KNeighborsClassifier(n_neighbors=20, weights='distance'),
        KNeighborsClassifier(n_neighbors=40, weights='distance'),
    ]

    # build classifier with clean learning
    base_clf_ = VotingClassifier(clfs, voting='soft')
    return base_clf_