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

from HAR.av.AVensembleNNPhase1.util import merge_dicts, jaccard_score_custom, aggregate_audio_ts_scores, fetch_model_data


def train_video_ensemble(train_raw_data,
                         video_model_list=('posec3d_hmdb', 'posec3d_ntu60', 'posec3d_ntu120', 'posec3d_ucf',
                                           'stgcn_ntu60'),
                         consideration_threshold=0.001,
                         consideration_label_count=10,
                         confidence_label_count=5,
                         model_confidence_percentile=50,
                         model_confidence_theshold_quantile=0.75,
                         clu_min_samples=3,
                         clu_max_eps=0.8,
                         uniformity_threshold=0.7,
                         ):
    """
    The train_video_ensemble function trains a video ensemble model for AV labels.

    Args:
        train_raw_data: Pass in the raw data from the training set
        video_model_list=('posec3d_hmdb': Specify the models used for video
        'posec3d_ntu60': Specify the model to use for posec3d
        'posec3d_ntu120': Specify the model that is used for video
        'posec3d_ucf': Specify the model to use for posec3d
        'stgcn_ntu60'): Specify the model name
        consideration_threshold=0.001: Filter out low-confidence predictions
        consideration_label_count=10: Set the number of top labels to consider for matching
        confidence_label_count=5: Determine the number of labels to consider for each model
        model_confidence_percentile=50: Select the percentile of model confidence for each activity
        model_confidence_threshold_quantile=0.75: Determine the threshold for confidence of a model in predicting an activity
        clu_min_samples=3: Determine the minimum number of samples in a cluster
        clu_max_eps=0.8: Set the maximum distance between two samples for one to be considered as in the neighborhood of the other
        uniformity_threshold=0.7: Determine if the activity is uniform across models
        : Determine the threshold for considering a label as present in an activity

    Returns:
        4 variables:

    Doc Author:
        Trelent
    """
    # get model confidence ptile for activities
    print("Training for video ensemble")
    model_activity_confidence = dict()
    for activity_name in train_raw_data.keys():
        for model in train_raw_data[activity_name].keys():
            if model not in video_model_list:
                continue
            input_dfs = train_raw_data[activity_name][model]
            if model in OpticsConfig.audio_models:
                input_dfs = [aggregate_audio_ts_scores(df_xr) for df_xr in input_dfs]
            ma_confidence = [xr.iloc[:confidence_label_count].values.sum() for xr in input_dfs]
            if activity_name not in model_activity_confidence:
                model_activity_confidence[activity_name] = {}
            model_activity_confidence[activity_name][model] = np.percentile(ma_confidence, model_confidence_percentile)

    df_model_confidence = pd.DataFrame.from_dict(model_activity_confidence)
    # df_model_confidence = None
    ## get model label uniformity
    model_activity_uniformity = dict()
    model_activity_clusters = dict()
    print("Getting Model Activity Clusters")
    for activity_name in train_raw_data.keys():
        for model in train_raw_data[activity_name].keys():
            if model not in video_model_list:
                continue
            input_dfs = train_raw_data[activity_name][model]
            if model in OpticsConfig.audio_models:
                input_dfs = [aggregate_audio_ts_scores(df_xr) for df_xr in input_dfs]
            top_labels = [xr[xr[1] > consideration_threshold].index[:consideration_label_count].values.tolist()
                          for xr in input_dfs]
            d_mat = np.zeros((len(input_dfs), len(input_dfs)))
            for i, lab_row in enumerate(top_labels):
                for j, lab_col in enumerate(top_labels):
                    d_mat[i][j] = 1 - jaccard_score_custom(lab_row, lab_col)
                    if (d_mat[i][j] == 0) & (not (i == j)): # penalize based on size of lab_row and lab_col with max distance be 0.2
                        d_mat[i][j] = 1e-5
                        # d_mat[i][j] = (consideration_label_count - len(lab_row))/5
            opclu = OPTICS(min_samples=min(clu_min_samples,d_mat.shape[0]), max_eps=clu_max_eps, metric='precomputed', n_jobs=-1)
            opclu.fit(d_mat)
            if activity_name not in model_activity_uniformity:
                model_activity_uniformity[activity_name] = {}
                model_activity_clusters[activity_name] = {}
            unique_label_counts = np.unique(opclu.labels_, return_counts=True)[1]
            model_activity_uniformity[activity_name][model] = 1 - (unique_label_counts[0] / np.sum(unique_label_counts))
            model_activity_clusters[activity_name][model] = (deepcopy(opclu), deepcopy(top_labels))

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
    #                     opclu = OPTICS(min_samples=clu_min_samples, max_eps=clu_max_eps, metric='precomputed', n_jobs=-1)
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


    df_model_uniformity = pd.DataFrame.from_dict(model_activity_uniformity)
    model_activity_match = np.copy(df_model_uniformity.values)
    model_activity_match[model_activity_match >= uniformity_threshold] = 1.
    model_activity_match[model_activity_match < uniformity_threshold] = 0.

    confidence_percentile_thresholds = df_model_confidence.quantile(model_confidence_theshold_quantile,
                                                                    axis=1).values.reshape(-1, 1)
    confidence_percentile_thresholds[confidence_percentile_thresholds > 0.9999] = 0.9999
    model_activity_match[
        (model_activity_match > 0.) & (df_model_confidence.values > confidence_percentile_thresholds)] = 1.
    df_model_match = pd.DataFrame(model_activity_match, index=df_model_confidence.index,
                                  columns=df_model_confidence.columns)
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
        'model_list': video_model_list
    }

    return trained_ensemble


def predict_from_video_ensemble(instance_dicts, trained_ensemble):
    """
    The predict_from_video_ensemble function takes a dictionary of test instances and a trained ensemble model.
    It returns two lists: predictions_all, which contains tuples of (test_instance_id, test_activity, prediction)
    and predictions detailed which contains tuples of (test instance id, ground truth activity label,
    predicted activity label with score). The function also accepts the following parameters: consideration threshold for
    the top labels in each video; number of top labels to consider; min samples for OPTICS clustering; max epsilon for OPTICS
    clustering.

    Args:
        instance_dicts: Pass the test instances
        trained_ensemble: Pass the trained ensemble model

    Returns:
        Two outputs:

    Doc Author:
        Trelent
    """
    print("Predictions from video ensemble")
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
            input_top_labels = input_df[input_df[1] > consideration_threshold].index[
                               :consideration_label_count].values.tolist()
            passed_activities = ma_match_all.columns[np.where(ma_match_all.loc[model, :].values > 0.)[0]]
            for activity in passed_activities:
                if model not in ma_clu[activity].keys():
                    instance_pred_detailed[model][activity] = 1-np.inf
                    continue
                train_clu, train_top_labels = ma_clu[activity][model]
                test_predictor_labels = [train_top_labels[xr] for xr in range(len(train_top_labels)) if
                                         train_clu.labels_[xr] >= 0]
                test_predictor_labels.append(input_top_labels)
                d_mat = np.zeros((len(test_predictor_labels), len(test_predictor_labels)))
                for i, lab_row in enumerate(test_predictor_labels):
                    for j, lab_col in enumerate(test_predictor_labels):
                        d_mat[i][j] = 1 - jaccard_score_custom(lab_row, lab_col)
                        if (d_mat[i][j] == 0) & (not (i == j)):
                            d_mat[i][j] = 1e-5
                opclu = OPTICS(min_samples=min(clu_min_samples,d_mat.shape[0]), max_eps=clu_max_eps, metric='precomputed')
                try:
                    opclu.fit(d_mat)
                    test_label, test_distance = opclu.labels_[-1], opclu.core_distances_[-1]
                except:
                    test_label, test_distance = -1, np.inf
                if (test_label >= 0) & (test_distance <= 1.):
                    #                     print(model, activity, input_top_labels, test_label, test_distance)
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
