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

from utils import time_diff


def combine_av_clustering(vax_pipeline_object, config, logger):
    """
    The combine_av_clustering function combines the raw audio and video labels with the clustering results.
    The function returns a dictionary of instance_id:prediction pairs, where prediction is a dictionary containing
    the final prediction and score for each sensor.

    Args:
        vax_pipeline_object: Get the raw_av_labels and sensor predictions
        config: Pass the configuration file to the function
        logger: Log the steps of the clustering process

    Returns:
        A dictionary with the instance_id as key and a dictionary of all the predictions as value

    Doc Author:
        Trelent
    """
    final_av_labels = dict()
    combine_av_clustering_ts_start = datetime.now()

    raw_video_output = vax_pipeline_object['raw_av_labels']['video']
    raw_audio_output = vax_pipeline_object['raw_av_labels']['audio']
    condensed_prediction = vax_pipeline_object['raw_av_labels']['condensed']


    # instances_cluster_info = vax_pipeline_object['training_clusters']
    sensor_predictions = vax_pipeline_object['sensor_predictions']

    # get prediction dataframe for all sensors
    sensor_predictions['condensed'] = condensed_prediction
    df_support_all = None
    for sensor in sensor_predictions:
        if not bool(sensor_predictions[sensor]):
            continue
        df_sensor_support_preds = pd.DataFrame.from_dict(sensor_predictions[sensor], orient='index')
        df_sensor_support_preds = pd.DataFrame(normalize(df_sensor_support_preds.fillna(0.).values, axis=1),
                                               index=df_sensor_support_preds.index,
                                               columns=df_sensor_support_preds.columns)
        sensor_support_preds = df_sensor_support_preds.idxmax(axis=1).reset_index()
        sensor_support_preds.columns = ['instance_id', f'{sensor}_prediction']
        sensor_support_scores = df_sensor_support_preds.max(axis=1).reset_index()
        sensor_support_scores.columns = ['instance_id', f'{sensor}_score']
        df_sensor_support = pd.merge(sensor_support_preds, sensor_support_scores, on=['instance_id'], how='outer')
        if df_support_all is None:
            df_support_all = df_sensor_support.copy(deep=True)
        else:
            df_support_all = pd.merge(df_support_all, df_sensor_support, on=['instance_id'], how='outer')

    df_support_all = df_support_all.set_index('instance_id')
    for instance_id in condensed_prediction.keys():
        if not bool(condensed_prediction[instance_id]):
            continue

        instance_prediction_dict = dict()

        # audio prediction and score
        df_pred_audio = raw_audio_output[instance_id]
        if df_pred_audio.shape[0]>0.:
            audio_prediction_score = np.nansum(df_pred_audio.values, axis=1).max()
            activity_scores = np.nansum(df_pred_audio.values, axis=1)
            audio_prediction = df_pred_audio.index[activity_scores.argmax()]
            instance_prediction_dict['audio_prediction'] = audio_prediction
            instance_prediction_dict['audio_score'] =audio_prediction_score
        else:
            instance_prediction_dict['audio_prediction'] = 'Undetected'
            instance_prediction_dict['audio_score'] = 0.

        # video prediction and score
        df_pred_video = raw_video_output[instance_id]
        if df_pred_video.shape[0]>0.:
            video_prediction_score = np.nansum(df_pred_video.values, axis=1).max()
            activity_scores = np.nansum(df_pred_video.values, axis=1)
            video_prediction = df_pred_video.index[activity_scores.argmax()]
            instance_prediction_dict['video_prediction'] = video_prediction
            instance_prediction_dict['video_score'] =video_prediction_score
        else:
            instance_prediction_dict['video_prediction'] = 'Undetected'
            instance_prediction_dict['video_score'] = 0.
        # condensed prediction and score


        # other sensors prediction
        if instance_id in df_support_all.index:
            instance_prediction_dict.update(df_support_all.loc[instance_id].to_dict())
            instance_final_prediction, instance_final_score = get_single_instance_prediction(config,
                                                                                             instance_prediction_dict)
            final_av_labels[instance_id] = instance_prediction_dict
            final_av_labels[instance_id].update({'final_prediction':instance_final_prediction, 'final_score':instance_final_score})

    combine_av_clustering_ts_end = datetime.now()

    logger.info(
        f"Got final av labels from raw av and clustering info in {time_diff(combine_av_clustering_ts_start, combine_av_clustering_ts_end)} secs.")

    return final_av_labels


def get_single_instance_prediction(config, instance_prediction_dict):
    row = instance_prediction_dict
    sensor_list = config['sensor_list']
    cutoffs = config['thresholds']

    if (row['audio_score'] > cutoffs['audio_high']):
        instance_final_pred, instance_final_score = row['audio_prediction'], row['audio_score']
    elif (row['audio_score'] > cutoffs['audio_low']) & (row['video_prediction'] == 'Undetected'):
        instance_final_pred, instance_final_score = row['audio_prediction'], row['audio_score']
    elif (row['video_score'] > cutoffs['video_high']):
        instance_final_pred, instance_final_score = row['video_prediction'], row['video_score']
    elif (row['video_score'] > cutoffs['video_low']) & (row['audio_prediction'] == 'Undetected'):
        instance_final_pred, instance_final_score = row['video_prediction'], row['video_score']
    else:
        instance_final_pred, instance_final_score = 'Undetected', 0.

    # if (not (row['condensed_prediction'] == instance_final_pred)) & (
    #         row['condensed_score'] > cutoffs['x_raw_av_min_score']):
    #     for sensor in sensor_list:
    #         if f'{sensor}_prediction' not in row.keys():
    #             continue
    #         if row[f'{sensor}_prediction']!=row[f'{sensor}_prediction']:# check for nan values
    #             continue
    #         if (row['condensed_prediction'] == row[f'{sensor}_prediction']) & (
    #                 row[f'{sensor}_score'] > cutoffs[f'x_{sensor}_min_score']):
    #             instance_final_pred, instance_final_score = row['condensed_prediction'], row['condensed_score']
    #             break

    return instance_final_pred, instance_final_score
