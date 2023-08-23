"""
Outut based on weighted sum across available sensor models
"""
import os
import pickle

import sklearn.preprocessing
import xgboost
import numpy as np
import pandas as pd

from sklearn.metrics import *
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.ensemble import *
from sklearn.model_selection import StratifiedKFold

activities = ['Baking', 'Blender', 'Chopping', 'CookingOnStove', 'Coughing', 'Drinking', 'Eating', 'Exercising',
              'FridgeOpen', 'Grating', 'HairBrush', 'HairDryer', 'HandWash', 'Knocking', 'Microwave', 'Shaver In Use',
              'Shower', 'Talking', 'ToilerFlushing', 'Toothbrush', 'Vacuum', 'Walking', 'WashingDishes', 'WatchingTV']

SCORE_ENSEMBLE_SENSOR_WEIGHTS = {
    'doppler': 1,
    'thermal': 1,
    'lidar': 1,
    'micarray': 1,
    'IMU': 0,
    'ENV': 0,
    'EMI': 0,
    'GridEye': 0,
    'PIR': 0,
    'WIFI': 0,
}


def getEnsembleOutput(instance_cv_output, allowed_sensors):
    """
    Get ensemble output at instance level cv
    :param instance_cv_output:
    :param allowed_sensors:
    :return:
    """
    all_ts = np.concatenate(
        [instance_cv_output[xr][0] for xr in allowed_sensors if ((instance_cv_output[xr][0] is not None))]).astype(int)
    all_gt = np.concatenate(
        [instance_cv_output[xr][2] for xr in allowed_sensors if (instance_cv_output[xr][2] is not None)])
    df_ts_gt = pd.DataFrame(np.array([all_ts, all_gt]).T, columns=['timestamp', 'GT']).drop_duplicates()
    df_ts_gt['timestamp'] = df_ts_gt['timestamp'].astype(int)
    min_ts, max_ts = all_ts.min(), all_ts.max()
    df_ts_pred = pd.DataFrame(index=np.arange(min_ts, max_ts + 1))
    df_ts_pred_prob = pd.DataFrame(index=np.arange(min_ts, max_ts + 1))
    df_ts_ensemble = pd.DataFrame(index=np.arange(min_ts, max_ts + 1))
    for sensor in allowed_sensors:
        df_sensor_pred = instance_cv_output[sensor][1]
        if df_sensor_pred is not None:
            df_sensor_ts = pd.merge(pd.DataFrame(index=np.arange(min_ts, max_ts + 1)), df_sensor_pred, left_index=True,
                                    right_index=True, how='left').fillna(0.)
            df_ts_pred[sensor] = df_sensor_ts.idxmax(axis=1, skipna=True)
            df_ts_pred_prob[sensor] = df_sensor_ts.max(axis=1, skipna=True)
            for col in df_sensor_ts.columns:
                if col not in df_ts_ensemble.columns:
                    df_ts_ensemble[col] = 0.
                df_ts_ensemble[col] += SCORE_ENSEMBLE_SENSOR_WEIGHTS[sensor] * df_sensor_ts[col]

    df_ts_pred['all'] = df_ts_ensemble.idxmax(axis=1, skipna=True)
    df_ts_pred_prob['all'] = df_ts_ensemble.max(axis=1, skipna=True)
    return df_ts_pred, df_ts_pred_prob, df_ts_gt
