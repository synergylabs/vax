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
from numpy.lib.stride_tricks import sliding_window_view



def get_features(raw_input, sensor):
    """
    The get_features function takes in a raw input and sensor type, and returns the time series of the features
    and the feature matrix. The featurizer is determined by featurize_module_map[featurizer]. For example, if
    featurizer='m2', then we will use m2's doppler module to get features from a Doppler sensor.

    Args:
        raw_input: Pass the raw data to the featurizer
        sensor: Specify the sensor type
        featurizer='m2': Select the featurizer module

    Returns:
        The time series and feature matrix for a given sensor

    Doc Author:
        Trelent
    """

    sensor_featurize = None
    if sensor=='doppler':
        sensor_featurize = doppler
    elif sensor=='lidar':
        sensor_featurize = lidar
    elif sensor=='thermal':
        sensor_featurize = thermal
    elif sensor=='micarray':
        sensor_featurize = micarray
    elif sensor=='IMU':
        sensor_featurize = IMU
    elif sensor=='EMI':
        sensor_featurize = EMI
    elif sensor=='ENV':
        sensor_featurize = ENV
    elif sensor=='PIR':
        sensor_featurize = PIR
    elif sensor=='GridEye':
        sensor_featurize = GridEye
    elif sensor=='WIFI':
        sensor_featurize = WIFI
    elif sensor=='FeatureMIC':
        sensor_featurize = FeatureMIC
    else:
        print("Sensor not available...")

    ts_sensor, X_sensor = sensor_featurize(raw_input)
    return ts_sensor, X_sensor

def doppler(instance_data, window_size=20):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and doppler data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    doppler_time_matrix = []
    for instance_ts, doppler_data_all in instance_data:
        if len(doppler_data_all.shape)==3:
            doppler_data = doppler_data_all[0]
        else:
            doppler_data = doppler_data_all
        noise_red_data = dsp.compensation.clutter_removal(doppler_data).T
        noise_red_data[noise_red_data < 0] = 0.
        doppler_time_matrix.append(noise_red_data)
    doppler_time_matrix = np.array(doppler_time_matrix)
    #reduce range information to closest 100 datapoints
    doppler_time_matrix = doppler_time_matrix[:,:,-100:]
    # print(doppler_time_matrix.shape)
    doppler_time_matrix = doppler_time_matrix[:,:,::-1]
    # get range matrix sliding over time
    rt_matrix = doppler_time_matrix.max(axis=1)
    noise_red_data = dsp.compensation.clutter_removal(rt_matrix)
    noise_red_data[noise_red_data < 0] = 0.
    if doppler_time_matrix.shape[0]>window_size:
        rt_data = sliding_window_view(noise_red_data, window_size, axis=0).sum(axis=2)
    else:
        rt_data = noise_red_data.sum(axis=0).reshape(1,-1)


    # get velocity matrix sliding over time
    vt_matrix_mean = doppler_time_matrix.mean(axis=2)
    vt_matrix_std = doppler_time_matrix.std(axis=2)
    vt_matrix = np.concatenate([vt_matrix_mean,vt_matrix_std],axis=1)
    # vt_matrix = doppler_time_matrix.std(axis=2)
    if doppler_time_matrix.shape[0]>window_size:
        vt_data = sliding_window_view(vt_matrix, window_size, axis=0).max(axis=2)
    else:
        vt_data = vt_matrix.sum(axis=0).reshape(1,-1)

    # aggregate time data
    if doppler_time_matrix.shape[0]>window_size:
        ts_data = sliding_window_view(timestamps,window_size,axis=0).max(axis=1)
    else:
        ts_data = np.array([timestamps.max()])
    ts_data = ts_data //1e9
    doppler_featurized_data = np.concatenate([rt_data,vt_data],axis=1)
    return ts_data, doppler_featurized_data


def lidar(instance_data, window_size=10):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and lidar data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    instant_feature_vec = []
    for instance_ts, lidar_data in instance_data:
        instant_feature_vec.append(lidar_data)

    instant_feature_vec = np.array(instant_feature_vec)
    instant_avg_vec = np.mean(instant_feature_vec, axis=0)
    boundary_mask = np.absolute(instant_avg_vec - instant_feature_vec) <= 100
    instant_feature_vec[boundary_mask] = 0.

    if instant_feature_vec.shape[0] > window_size:
        lidar_data = sliding_window_view(instant_feature_vec, window_size, axis=0).max(axis=2)
    else:
        lidar_data = instant_feature_vec.sum(axis=0).reshape(1,-1)

    if instant_feature_vec.shape[0] > window_size:
        ts_data = sliding_window_view(timestamps,window_size,axis=0).max(axis=1)
    else:
        ts_data = np.array([timestamps.max()])
    ts_data = ts_data // 1e9

    return ts_data, lidar_data



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
        micarray_data_sst, micarray_data_ssl = micarray_data
        # Method 1: ssts only
        instant_feature_vec.append(micarray_data_sst)
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


def thermal(instance_data, thermal_compression_=16, window_size=5):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and thermal data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, thermal_data in instance_data:
        instant_feature_vec.append(thermal_data.flatten())
    instant_feature_vec = np.array(instant_feature_vec)

    if instant_feature_vec.shape[0] > window_size:
        thermal_data = sliding_window_view(instant_feature_vec, window_size, axis=0).max(axis=2)
    else:
        thermal_data = instant_feature_vec.sum(axis=0).reshape(1,-1)

    if instant_feature_vec.shape[0] > window_size:
        ts_data = sliding_window_view(timestamps,window_size,axis=0).max(axis=1)
    else:
        ts_data = np.array([timestamps.max()])
    ts_data = ts_data // 1e9
    return ts_data, thermal_data


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

def FeatureMIC(instance_df):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and mites data
    :return:
    """

    instance_df = instance_df.drop(['TimeStamp', 'ts'], axis=1)
    accepted_columns = ['UnixTimestamp'] + sorted(
        [cl for cl in instance_df.columns if ('Mic' in cl)])
    instance_df = instance_df[accepted_columns]
    instance_df_arr = instance_df.groupby('UnixTimestamp').mean()
    return instance_df_arr.index.values, instance_df_arr.values


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
