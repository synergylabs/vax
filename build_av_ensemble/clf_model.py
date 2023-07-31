"""
This is main function to train AV model ensemble
"""
from collections import Counter
import numpy as np
import pandas as pd
from HAR.av.AVensembleFixedWindow.config import OpticsConfig
import glob
import pickle
from copy import deepcopy
import time
from sklearn.cluster import OPTICS
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import DistanceMetric
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def train_clf_ensemble(train_raw_data,
                       clf_,
                       gridsearch_param_grid,
                       model_list=('yamnet', 'posec3d_hmdb', 'posec3d_ntu120', 'posec3d_ucf'),
                       cv_folds=5,
                       top_label_count=10,
                       clu_max_eps=0.5,

                       ):
    model_activity_datasets = dict()
    print("Getting Model Activity Clusters")
    start_time = time.time()
    # get dataset for prediction from simple classification model
    for activity_name in train_raw_data.keys():
        model_activity_datasets[activity_name] = {}
        for model in train_raw_data[activity_name].keys():
            # print(f"Collecting data for {model} for {activity_name}")
            if model not in model_list:
                continue
            instance_dfs = [train_raw_data[activity_name][model]]
            top_label_dfs = []
            for i in range(len(instance_dfs)):
                top_label_dfs += [xr.iloc[:top_label_count] for xr in instance_dfs[i]]

            df_X_trainer = pd.concat(top_label_dfs, axis=1).sort_index().transpose().fillna(0.)
            df_X_trainer['vax_activity_label'] = activity_name
            model_activity_datasets[activity_name][model] = df_X_trainer

    # train classification model
    model_classifiers = dict()
    for model_name in model_list:
        model_start_time = time.time()
        df_model_data = pd.concat([model_activity_datasets[xr][model_name] for xr in train_raw_data.keys() if
                                   model_name in model_activity_datasets[xr]], ignore_index=True)
        df_X = df_model_data.drop('vax_activity_label', axis=1).fillna(0.)
        X = df_X.values
        y_str = df_model_data['vax_activity_label'].values
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y_str)
        clf_gridsearch_ = GridSearchCV(estimator=clf_, param_grid=gridsearch_param_grid,
                                       refit=True, cv=cv_folds)
        clf_gridsearch_.fit(X,y_enc)
        model_clf_ = clf_gridsearch_.best_estimator_
        X_dummy_tester = pd.DataFrame(np.zeros(df_X.shape[1]), index=df_X.columns).transpose()
        print(f"Trained classifier for model {model_name} with best score {clf_gridsearch_.best_score_:3f},{str(clf_gridsearch_.best_params_)} in {time.time()-model_start_time:3f} secs..")
        model_classifiers[model_name] = (deepcopy(model_clf_), deepcopy(label_encoder), deepcopy(X_dummy_tester))

    trained_ensemble = {
        'model_activity_dataset': model_activity_datasets,
        'model_classifiers': model_classifiers,
        'clu_max_eps': clu_max_eps,
        'model_list': model_list,
    }
    print(f"Total training time: {time.time() - start_time:3f} secs...")

    return trained_ensemble


def predict_from_clf_ensemble(instance_dicts, trained_ensemble):
    print("Predictions from ensemble")
    predictions_detailed = {}
    predictions_all = []
    model_classifiers = trained_ensemble['model_classifiers']
    model_list = trained_ensemble['model_list']

    for test_instance_id in instance_dicts:
        start_time = time.time()
        test_instance = instance_dicts[test_instance_id]
        test_activity = test_instance['activity_gt']
        instance_pred_detailed = {}
        window_pred_detailed = {}
        for model in test_instance:
            if model not in model_list:
                continue
            instance_pred_detailed[model] = {}
            window_pred_detailed[model] = {}
            input_dfs = [test_instance[model]]
            model_clf_, model_enc_, dummy_test_df = model_classifiers[model]
            for input_df in input_dfs:
                for label in input_df.index:
                    if label in dummy_test_df.columns:
                        dummy_test_df[label] = input_df.loc[label, 1]
                if np.sum(dummy_test_df.values) > 0:
                    test_probs = model_clf_.predict_proba(dummy_test_df.values.reshape(1, -1))[0]
                    for y_lab, y_prob in enumerate(test_probs):
                        test_label = model_enc_.inverse_transform([y_lab])[0]
                        instance_pred_detailed[model][test_label] = max(instance_pred_detailed[model].get(test_label, 0.),
                                                                        y_prob)

        df_instance_pred = pd.DataFrame.from_dict(instance_pred_detailed).fillna(0.)
        df_window_pred = pd.DataFrame.from_dict(window_pred_detailed, dtype=object)
        df_instance_pred = df_instance_pred[
            df_instance_pred.index.isin(
                OpticsConfig.context_activities[OpticsConfig.activity_context_map[test_activity]])]
        df_window_pred = df_window_pred[
            df_window_pred.index.isin(
                OpticsConfig.context_activities[OpticsConfig.activity_context_map[test_activity]])]
        if df_instance_pred.shape[0] > 0.:
            max_prediction_score = np.nansum(df_instance_pred.values, axis=1).max()
            activity_scores = np.nansum(df_instance_pred.values, axis=1)
            prediction = df_instance_pred.index[activity_scores.argmax()]
            predictions_all.append((test_instance_id, test_activity, prediction, max_prediction_score))
        else:
            prediction = 'Undetected'
            max_prediction_score = 0.
            predictions_all.append((test_instance_id, test_activity, 'Undetected', 0.))
        # print(predictions_all[-1])
        predictions_detailed[test_instance_id] = (df_instance_pred, df_window_pred, test_activity)
        end_time = time.time()
        # print(f"Predicted {test_instance_id}:{test_activity},{prediction}:{max_prediction_score} in {end_time - start_time:2f} secs..")
    return predictions_all, predictions_detailed
