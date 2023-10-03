'''

'''

import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from collections import Counter
import os
import pickle
from copy import deepcopy
import shutil
# mmwave for noise reduction
# import mmwave.dsp as dsp
# import mmwave.clustering as clu
import itertools

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
import xgboost
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture

from datetime import datetime

from utils import time_diff, merge_dicts, jaccard_score_custom, aggregate_ts_scores


def get_raw_av_labels(vax_pipeline_object, config, logger):
    raw_av_labels = dict()
    raw_av_labels_ts_start = datetime.now()

    window_length = config["instance_window_length"]
    raw_av_labels_cache_file = f'{config["cache_dir"]}/raw_av_labels_{window_length}.pb'

    if os.path.exists(raw_av_labels_cache_file):
        logger.info("Got Raw AV labels from cache file...")
        raw_av_labels = pickle.load(open(raw_av_labels_cache_file, 'rb'))
    else:
        raw_av_instances_cache_dir = f'{config["cache_dir"]}/raw_av_instances'
        if not os.path.exists(raw_av_instances_cache_dir):
            os.makedirs(raw_av_instances_cache_dir)


        otc_labels = vax_pipeline_object['otc_labels']

        # get av_ensemble from reference homes
        av_ensemble_file = config["av_ensemble_file"]
        _, video_ensemble, audio_ensemble = pickle.load(open(av_ensemble_file, 'rb'))

        # get predictions from video ensemble
        video_predictions = dict()
        for instance_id in otc_labels:
            video_pred_instance_cache = f'{raw_av_instances_cache_dir}/video_{instance_id}.csv'
            if os.path.exists(video_pred_instance_cache):
                df_instance_pred = pd.read_csv(video_pred_instance_cache,index_col=0)
            else:
                model_predictions = dict()
                for model_name in config['video_models']:
                    if model_name in otc_labels[instance_id]:
                        df_otc_output = otc_labels[instance_id][model_name]
                        model_activity_predictions = get_video_ensemble_prediction(config, video_ensemble, model_name,
                                                                                   df_otc_output)
                        model_predictions[model_name] = model_activity_predictions
                df_instance_pred = pd.DataFrame.from_dict(model_predictions)
                df_instance_pred.to_csv(video_pred_instance_cache)
            video_predictions[instance_id] = df_instance_pred
        raw_av_labels['video'] = video_predictions

        # get predictions from audio ensemble
        audio_predictions = dict()
        for instance_id in otc_labels:
            audio_pred_instance_cache = f'{raw_av_instances_cache_dir}/audio_{instance_id}.csv'
            if os.path.exists(audio_pred_instance_cache):
                df_instance_pred = pd.read_csv(audio_pred_instance_cache,index_col=0)
            else:
                model_predictions = dict()
                for model_name in config['audio_models']:
                    if model_name in otc_labels[instance_id]:
                        df_otc_output = otc_labels[instance_id][model_name]
                        model_activity_predictions = get_audio_ensemble_prediction(config, audio_ensemble, model_name,
                                                                                   df_otc_output)
                        model_predictions[model_name] = model_activity_predictions
                df_instance_pred = pd.DataFrame.from_dict(model_predictions)
                df_instance_pred.to_csv(audio_pred_instance_cache)
            audio_predictions[instance_id] = df_instance_pred
        raw_av_labels['audio'] = audio_predictions

        # get condensed predictions from audio and video
        video_condensed_prediction = {
            instance_id: (raw_av_labels['video'][instance_id].max(axis=1) / raw_av_labels['video'][instance_id].max(
                axis=1).sum()).to_dict() for
            instance_id in raw_av_labels['video'].keys()}
        audio_condensed_prediction = {
            instance_id: (raw_av_labels['audio'][instance_id].max(axis=1) / raw_av_labels['audio'][instance_id].max(
                axis=1).sum()).to_dict() for
            instance_id in raw_av_labels['audio'].keys()}
        condensed_prediction = deepcopy(video_condensed_prediction)
        for instance_id in audio_condensed_prediction:
            if instance_id in condensed_prediction:
                condensed_prediction[instance_id] = merge_dicts(
                    [condensed_prediction[instance_id], audio_condensed_prediction[instance_id]])

        raw_av_labels['condensed'] = condensed_prediction
        pickle.dump(raw_av_labels, open(raw_av_labels_cache_file, 'wb'))
    raw_av_labels_ts_end = datetime.now()
    logger.info(
        f"Got raw A/V labels in {time_diff(raw_av_labels_ts_start, raw_av_labels_ts_end)}")

    return raw_av_labels


def get_video_ensemble_prediction(config, trained_ensemble, model_name, df_otc_output):
    # consideration_threshold = trained_ensemble['consideration_threshold']
    # consideration_label_count = trained_ensemble['consideration_label_count']
    # clu_min_samples = trained_ensemble['clu_min_samples']
    # clu_max_eps = trained_ensemble['clu_max_eps']
    # ma_clu = trained_ensemble['model_activity_cluster']
    model_list = trained_ensemble['model_list']
    if model_name not in model_list:
        return None  # todo: raise error in future

    df_otc_output = df_otc_output[~df_otc_output.index.isin(config['label_filters'][model_name])]
    df_out = aggregate_ts_scores(df_otc_output)
    if model_name == 'stgcn':
        df_out = df_out[df_out[1] > 0.].sort_values(by=[1], ascending=False)
        df_out[1] = np.exp(df_out[1]) / np.sum(np.exp(df_out[1]))

    input_dfs = [df_out]
    model_clf_, model_enc_, dummy_test_df = trained_ensemble['model_classifiers'][model_name]
    model_activity_predictions = dict()
    for input_df in input_dfs:
        for label in input_df.index:
            if label in dummy_test_df.columns:
                dummy_test_df[label] = input_df.loc[label, 1]
        if np.sum(dummy_test_df.values) > 0:
            test_probs = model_clf_.predict_proba(dummy_test_df.values.reshape(1, -1))[0]
            for y_lab, y_prob in enumerate(test_probs):
                test_label = model_enc_.inverse_transform([y_lab])[0]
                if test_label in config['activity_set']:
                    model_activity_predictions[test_label] = max(model_activity_predictions.get(test_label, 0.),
                                                                    y_prob)
    return model_activity_predictions


def get_audio_ensemble_prediction(config, trained_ensemble, model_name, df_otc_output):
    # consideration_threshold = trained_ensemble['consideration_threshold']
    # consideration_label_count = trained_ensemble['consideration_label_count']
    # clu_min_samples = trained_ensemble['clu_min_samples']
    # clu_max_eps = trained_ensemble['clu_max_eps']
    # ma_clu = trained_ensemble['model_activity_cluster']
    model_list = trained_ensemble['model_list']
    # distance_metric = trained_ensemble['distance_metric']
    if model_name not in model_list:
        return None  # todo: raise error in future

    df_otc_output = df_otc_output[~df_otc_output.index.isin(config['label_filters'][model_name])]
    df_out = aggregate_ts_scores(df_otc_output)
    if model_name == 'stgcn':
        df_out = df_out[df_out[1] > 0.].sort_values(by=[1], ascending=False)
        df_out[1] = np.exp(df_out[1]) / np.sum(np.exp(df_out[1]))

    input_dfs = [df_out]
    model_clf_, model_enc_, dummy_test_df = trained_ensemble['model_classifiers'][model_name]
    model_activity_predictions = dict()
    for input_df in input_dfs:
        for label in input_df.index:
            if label in dummy_test_df.columns:
                dummy_test_df[label] = input_df.loc[label, 1]
        if np.sum(dummy_test_df.values) > 0:
            test_probs = model_clf_.predict_proba(dummy_test_df.values.reshape(1, -1))[0]
            for y_lab, y_prob in enumerate(test_probs):
                test_label = model_enc_.inverse_transform([y_lab])[0]
                if test_label in config['activity_set']:
                    model_activity_predictions[test_label] = max(model_activity_predictions.get(test_label, 0.),
                                                                 y_prob)
    return model_activity_predictions
