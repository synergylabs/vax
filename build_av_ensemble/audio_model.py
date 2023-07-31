"""
This is main function to train AV model ensemble
"""
from collections import Counter
import numpy as np
import pandas as pd
from HAR.av.AVensemble.config import OpticsConfig
import glob
import pickle
from copy import deepcopy
from sklearn.cluster import OPTICS
import warnings

warnings.filterwarnings("ignore")

from build_av_ensemble.util import merge_dicts, jaccard_score_custom, aggregate_audio_ts_scores, fetch_model_data


def train_audio_ensemble(train_raw_data,
                      audio_model_list=('yamnet',),
                      confidence_label_count=5,
                      model_confidence_percentile=50,
                      model_confidence_theshold_quantile=0.8,
                      consideration_threshold=0.001,
                      consideration_label_count=5,
                      clu_min_samples=10,
                      clu_max_eps=0.5,
                      uniformity_threshold=0.6,
                      distance_metric='cosine'
                      ):
    """
    The train_audio_model function trains an audio model based on the provided training data.
    The function returns a trained ensemble of models that can be used for prediction.


    Args:
        train_raw_data: Train the model
        audio_model_list=('yamnet'): Specify which models to use for the training
        confidence_label_count=5: Determine the number of labels that will be used to train the model
        model_confidence_percentile=50: Determine the percentile of the confidence scores for each activity
        model_confidence_theshold_quantile=0.8: Determine the confidence threshold for each model
        consideration_threshold=0.001: Filter out low confidence predictions
        consideration_label_count=5: Determine how many labels to consider when determining the confidence of a model
        clu_min_samples=10: Determine the minimum number of samples required to be considered a cluster
        clu_max_eps=0.5: Determine the maximum distance between two points for them to be considered as in the same neighborhood
        uniformity_threshold=0.6: Determine the minimum uniformity of the clusters
        distance_metric='cosine': Define the distance metric used in optics clustering

    Returns:
        A trained ensemble model that can be used to predict activities

    Doc Author:
        Trelent
    """
    # get model confidence ptile for activities
    print("Training for audio ensemble")
    model_activity_confidence = dict()
    for activity_name in train_raw_data.keys():
        for model in train_raw_data[activity_name].keys():
            if model not in audio_model_list:
                continue
            input_dfs = train_raw_data[activity_name][model]
            if model in OpticsConfig.audio_models:
                input_dfs = [aggregate_audio_ts_scores(df_xr) for df_xr in input_dfs]
            ma_confidence = [xr.iloc[:confidence_label_count].values.sum() for xr in input_dfs]
            if activity_name not in model_activity_confidence:
                model_activity_confidence[activity_name] = {}
            model_activity_confidence[activity_name][model] = np.percentile(ma_confidence, model_confidence_percentile)

    df_model_confidence = pd.DataFrame.from_dict(model_activity_confidence)

    ## get model label uniformity
    model_activity_uniformity = dict()
    model_activity_clusters = dict()
    print("Getting Model Activity Clusters")
    for activity_name in train_raw_data.keys():
        for model in train_raw_data[activity_name].keys():
            if model not in audio_model_list:
                continue
            input_dfs = train_raw_data[activity_name][model]
            input_dfs = [aggregate_audio_ts_scores(df_xr) for df_xr in input_dfs]
            top_label_dfs = [xr[xr[1] > consideration_threshold].iloc[:consideration_label_count]
                             for xr in input_dfs]
            X_trainer = pd.concat(top_label_dfs, axis=1).sort_index().transpose().fillna(0.).values

            # cluster trainer for given model and activity
            opclu = OPTICS(min_samples=min(clu_min_samples,X_trainer.shape[0]), max_eps=clu_max_eps, metric=distance_metric, n_jobs=-1)
            opclu.fit(X_trainer)
            if activity_name not in model_activity_uniformity:
                model_activity_uniformity[activity_name] = {}
                model_activity_clusters[activity_name] = {}
            unique_label_counts = np.unique(opclu.labels_, return_counts=True)[1]
            model_activity_uniformity[activity_name][model] = 1 - (unique_label_counts[0] / np.sum(unique_label_counts))
            model_activity_clusters[activity_name][model] = (deepcopy(opclu), deepcopy(top_label_dfs))

    # print("Getting Separability Matrix...")
    # # get separability metrics
    model_activity_separability = dict()
    # for activity_name in model_activity_clusters.keys():
    #     for model in model_activity_clusters[activity_name].keys():
    #         train_clu, train_top_labels = model_activity_clusters[activity_name][model]
    #         for ref_activity_name in model_activity_clusters.keys():
    #             if ref_activity_name not in OpticsConfig.context_activities[OpticsConfig.activity_context_map[activity_name]]:
    #                 continue
    #             if not (ref_activity_name==activity_name):
    #                 ref_overlap = []
    #                 for tr_data in model_activity_clusters[ref_activity_name][model][1]:
    #                     test_predictor_labels = list(train_top_labels)
    #                     test_predictor_labels.append(tr_data)
    #                     d_mat = np.zeros((len(test_predictor_labels), len(test_predictor_labels)))
    #                     for i, lab_row in enumerate(test_predictor_labels):
    #                         for j, lab_col in enumerate(test_predictor_labels):
    #                             d_mat[i][j] = 1 - jaccard_score_custom(lab_row, lab_col)
    #                             if (d_mat[i][j] == 0) & (not (i == j)):
    #                                 d_mat[i][j] = 1e-5
    #                     opclu = OPTICS(min_samples=clu_min_samples, max_eps=clu_max_eps, metric=distance_metric, n_jobs=-1)
    #                     try:
    #                         opclu.fit(d_mat)
    #                         test_label, test_distance = opclu.labels_[-1], opclu.core_distances_[-1]
    #                     except:
    #                         test_label, test_distance = -1, 1.
    #                     ref_overlap.append(1 - test_distance)
    #                 model_activity_separability[tuple([activity_name,model,ref_activity_name])] =ref_overlap
    #                 print(f"{activity_name},{model},{ref_activity_name}")
    #         print(f"Got Separability array for {activity_name},{model}")
    #     print(f"==Got Separability matrix for {activity_name}==")



    # Create model match based on uniformity and confidence
    df_model_uniformity = pd.DataFrame.from_dict(model_activity_uniformity)
    model_activity_match = np.copy(df_model_uniformity.values)
    model_activity_match[model_activity_match >= uniformity_threshold] = 1.
    model_activity_match[model_activity_match < uniformity_threshold] = 0.

    confidence_percentile_thresholds = df_model_confidence.quantile(model_confidence_theshold_quantile,
                                                                    axis=1).values.reshape(-1, 1)
    model_activity_match[(df_model_confidence.values >= confidence_percentile_thresholds)] = 1.
    df_model_match = pd.DataFrame(model_activity_match, index=df_model_confidence.index,
                                  columns=df_model_confidence.columns)

    # create trained ensemble for independent prediction
    trained_ensemble = {
        'model_match': df_model_match,
        'model_activity_cluster': model_activity_clusters,
        'model_activity_separability': model_activity_separability,
        'model_confidence': df_model_confidence,
        'model_uniformity': df_model_uniformity,
        'consideration_threshold': consideration_threshold,
        'consideration_label_count': consideration_label_count,
        'confidence_label_count': confidence_label_count,
        'model_confidence_percentile': model_confidence_percentile,
        'model_confidence_threshold_quantile': model_confidence_theshold_quantile,
        'clu_min_samples': clu_min_samples,
        'clu_max_eps': clu_max_eps,
        'uniformity_threshold': uniformity_threshold,
        'model_list': audio_model_list,
        'distance_metric': distance_metric
    }

    return trained_ensemble


def predict_from_audio_ensemble(instance_dicts, trained_ensemble):
    """
    The predict_from_audio_model function takes in a dictionary of test instances and the trained ensemble model.
    It returns two lists: predictions_all, which contains tuples of (test instance id, ground truth activity label,
    predicted activity label) for each test instance; and predictions_detailed, which contains a tuple containing
    the prediction scores for each activity in the form (df_instance_predictions, ground truth activity label). The
    second list is returned because it can be used to visualize how well the model performed on individual instances.

    Args:
        instance_dicts: Pass the dictionary of test instances to the function
        trained_ensemble: Specify the ensemble model that is used to make predictions

    Returns:
        Two objects:

    Doc Author:
        Trelent
    """
    print("Predictions from audio ensemble")
    predictions_detailed = {}
    predictions_all = []
    consideration_threshold = trained_ensemble['consideration_threshold']
    consideration_label_count = trained_ensemble['consideration_label_count']
    clu_min_samples = trained_ensemble['clu_min_samples']
    clu_max_eps = trained_ensemble['clu_max_eps']
    ma_match = trained_ensemble['model_match']
    ma_match_all = pd.DataFrame(np.ones_like(ma_match), index=ma_match.index, columns=ma_match.columns)
    ma_clu = trained_ensemble['model_activity_cluster']
    ma_conf = trained_ensemble['model_confidence']
    model_list = trained_ensemble['model_list']
    distance_metric = trained_ensemble['distance_metric']

    for test_instance_id in instance_dicts:
        test_instance = instance_dicts[test_instance_id]
        test_activity = test_instance['activity_gt']
        instance_pred_detailed = {}
        for model in test_instance:
            if model not in model_list:
                continue
            instance_pred_detailed[model] = {}
            input_df = test_instance[model]
            if model in OpticsConfig.audio_models:
                input_df = aggregate_audio_ts_scores(input_df)
            passed_activities = ma_match_all.columns[np.where(ma_match_all.loc[model, :].values > 0.)[0]]
            for activity in passed_activities:
                train_clu, top_label_dfs = ma_clu[activity][model]
                test_predictor_dfs = [top_label_dfs[xr] for xr in range(len(top_label_dfs)) if
                                      train_clu.labels_[xr] >= 0]
                test_predictor_dfs.append(input_df)
                X_test_predictor = pd.concat(test_predictor_dfs, axis=1).sort_index().transpose().fillna(
                    0.).values
                opclu = OPTICS(min_samples=min(clu_min_samples,X_test_predictor.shape[0]), max_eps=clu_max_eps, metric=distance_metric)
                try:
                    opclu.fit(X_test_predictor)
                    test_label, test_distance = opclu.labels_[-1], opclu.core_distances_[-1]
                except:
                    test_label, test_distance = -1, np.inf
                if (test_label >= 0) & (test_distance <= 1.):
                    instance_pred_detailed[model][activity] = 1 - test_distance
        df_instance_pred = pd.DataFrame.from_dict(instance_pred_detailed)
        df_instance_pred = df_instance_pred[
            df_instance_pred.index.isin(
                OpticsConfig.context_activities[OpticsConfig.activity_context_map[test_activity]])]
        if df_instance_pred.shape[0] > 0.:
            max_prediction_score = np.nansum(df_instance_pred.values, axis=1).max()
            activity_scores = np.nansum(df_instance_pred.values, axis=1)
            if len(activity_scores[activity_scores == np.nanmax(activity_scores)]) > 1.:
                pred_indexes = df_instance_pred.index[
                    np.nansum(df_instance_pred.values, axis=1) >= max_prediction_score]
                mdc = ma_conf.loc[df_instance_pred.columns, pred_indexes].transpose()
                mdc[df_instance_pred.isnull()] = np.nan
                mdc = mdc * df_instance_pred.loc[pred_indexes]
                prediction = mdc.index[np.nansum(mdc.values, axis=1).argmax()]
                predictions_all.append((test_instance_id, test_activity, prediction, max_prediction_score))
                # print("Multiple output:", pred_indexes, test_activity, prediction)
            else:
                prediction = df_instance_pred.index[activity_scores.argmax()]
                predictions_all.append((test_instance_id, test_activity, prediction, max_prediction_score))
                # print(predictions_all[-1])
        else:
            predictions_all.append((test_instance_id, test_activity, 'Undetected', 0.))
        predictions_detailed[test_instance_id] = (df_instance_pred, test_activity)
    return predictions_all, predictions_detailed
