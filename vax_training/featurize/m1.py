"""
Featurization for multiple functions
"""

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

# mmwave for noise reduction
import mmwave.dsp as dsp
import mmwave.clustering as clu

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
import xgboost
from sklearn.svm import SVC
from sklearn.model_selection import *
import skimage.measure


def doppler(instance_data):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and doppler data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, doppler_data in instance_data:
        if isinstance(doppler_data,dict):
            doppler_data = doppler_data['det_matrix']
        noise_red_data = dsp.compensation.clutter_removal(doppler_data).T
        noise_red_data[noise_red_data < 0] = 0.
        range_vec = np.max(noise_red_data, axis=0)
        doppler_vec = np.max(noise_red_data, axis=1)
        instant_feature_vec.append(np.concatenate([range_vec, doppler_vec]))

    instant_feature_vec = np.array(instant_feature_vec)
    sec_ts = []
    sec_feature_vec = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.median(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


def lidar(instance_data):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and lidar data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, lidar_data in instance_data:
        lidar_data_x, lidar_data_y = lidar_data
        lidar_data_x, lidar_data_y = np.array(lidar_data_x), np.array(lidar_data_y)
        lidar_data_r = np.sqrt((lidar_data_x ** 2) + (lidar_data_y ** 2))
        lidar_data_th = np.arctan2(lidar_data_y, lidar_data_x) * 180 / np.pi
        sort_indices = np.argsort(lidar_data_th)
        lidar_data_th = lidar_data_th[sort_indices]
        lidar_data_r = lidar_data_r[sort_indices]
        bin_count, _ = np.histogram(lidar_data_th, range(-180, 180))
        bin_idx = np.digitize(lidar_data_th, range(-180, 180)) - 180
        df_inst = pd.DataFrame(np.array([bin_idx, lidar_data_r]).T, columns=['angle', 'distance'])
        # remove less than 200 cms to remove corners and rig interference
        # df_inst = df_inst[df_inst.distance>200]
        df_inst = df_inst.groupby('angle', as_index=False).agg({'distance': 'mean'})
        df_inst = df_inst.astype(int)
        # df_inst['distance'] = (df_inst['distance']//100)*100
        df_inst = pd.merge(pd.DataFrame(range(-180, 180), columns=['angle']), df_inst[['angle', 'distance']],
                           how='left').fillna(0.)
        instant_feature_vec.append(df_inst.distance.values)

    instant_feature_vec = np.array(instant_feature_vec)
    instant_avg_vec = np.mean(instant_feature_vec, axis=0)
    boundary_mask = np.absolute(instant_avg_vec - instant_feature_vec) <= 100
    instant_feature_vec[boundary_mask] = 0.
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


def cyl2sph(x, y, z):
    # z = rcos(phi)
    # x = rsin(phi)cos(theta)
    # y = rsin(phi)sin(theta)
    if x == y == z == 0:
        return 0, np.nan, np.nan
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    theta_sign = np.sign(np.arcsin(y / (r * np.sin(phi))))
    theta_ = np.arccos(x / (r * np.sin(phi)))
    return r, phi * 180 / np.pi, theta_sign * theta_ * 180 / np.pi


def featurize_ss(ss_arr):
    res = np.zeros(49)
    res[-1] = ss_arr[4]
    r, phi, theta = cyl2sph(ss_arr[1], ss_arr[2], ss_arr[3])
    if r > 0:
        # localize in 45 degree cones
        flat_cone_idx = np.digitize([theta], bins=range(-180, 180, 45))
        z_cone_idx = np.digitize([phi], bins=range(0, 180, 45))
        cone_idx = (4 * flat_cone_idx) + z_cone_idx
        res[cone_idx] = r
    return res


def micarray(instance_data):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and micarray data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, micarray_data in instance_data:
        micarray_data_sst, micarray_data_ssl = micarray_data['SST'].values(), micarray_data['SSL'].values()
        feature_ssts = np.array([featurize_ss(xr) for xr in micarray_data_sst])
        # Method 1: ssts only
        instant_feature_vec.append(np.sum(feature_ssts, axis=0))
        # # Method 2: combine ssls and ssts
        # feature_ssls = np.array([featurize_ss([0]+xr) for xr in micarray_data_ssl])
        # instant_feature_vec.append(np.sum(feature_ssts+(0.25*feature_ssls),axis=0))
    instant_feature_vec = np.array(instant_feature_vec)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


def thermal(instance_data, thermal_compression_=16):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and thermal data
    :return:
    """
    instance_data_compressed = []
    for i in range(len(instance_data)):
        instance_data_compressed.append(
            (instance_data[i][0], skimage.measure.block_reduce(instance_data[i][1], (
                thermal_compression_, thermal_compression_), np.max)))

    timestamps = np.array([xr[0] for xr in instance_data_compressed])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, thermal_data in instance_data_compressed:
        instant_feature_vec.append(thermal_data.flatten())

    instant_feature_vec = np.array(instant_feature_vec)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        mean_instance_vals = np.median(instant_feature_vec[timestamp_secs == ts], axis=0)
        std_instance_vals = np.std(instant_feature_vec[timestamp_secs == ts], axis=0)
        sec_feature_vec.append(np.concatenate([mean_instance_vals, std_instance_vals]))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


# def mites(instance_df):
#     """
#     Returns featurization based on max in sample and median across sample at 1 sec level
#     :param instance_data: list of tuples with ts and mites data
#     :return:
#     """
#     instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
#     accepted_columns = [cl for cl in instance_df.columns if ('Mic' not in cl)]
#     instance_df = instance_df[accepted_columns]
#     instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
#     return instance_df_arr.index.values, instance_df_arr.values


def IMU(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted(
        [cl for cl in instance_df.columns if (('Accel' in cl) | ('Mag' in cl))])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


def GridEye(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted([cl for cl in instance_df.columns if ('GridEye' in cl)])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


def EMI(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted([cl for cl in instance_df.columns if ('EMI' in cl)])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


def PIR(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted([cl for cl in instance_df.columns if ('Motion' in cl)])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


def ENV(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted([cl for cl in instance_df.columns if
                                                   (('Temp' in cl) | ('Baro' in cl) | ('Hum' in cl) | (
                                                           'Light' in cl) | ('Color' in cl))])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


def WIFI(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted([cl for cl in instance_df.columns if (('WiFi' in cl))])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values
