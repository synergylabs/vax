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
from featurize.m1 import get_features


def featurize_training_data(vax_pipeline_object, config, logger):
    featurized_training_data = dict()
    featurize_training_data_ts_start = datetime.now()

    featurized_sensor_data_cache = f'{config["cache_dir"]}/featurized_sensor_data.pb'
    if os.path.exists(featurized_sensor_data_cache):
        featurized_training_data = pickle.load(open(featurized_sensor_data_cache, 'rb'))
    else:
        featurized_training_data = dict()
        for instance_id in vax_pipeline_object['instances']:
            featurized_training_data[instance_id] = {'instance_id': instance_id}
            # instance_dir = vax_pipeline_object['instances'][instance_id]['instance_dir']
            instance_dir = f'{config["processed_data_dir"]}/{instance_id}'
            for sensor_idx, sensor in enumerate(config['sensor_list']):
                sensor_filepath = f"{instance_dir}/{config['sensor_files'][sensor_idx]}"
                ts_sensor, X_sensor = None, None
                if os.path.exists(sensor_filepath):
                    if sensor_filepath.split(".")[-1] == 'pb':
                        raw_sensor_data = pickle.load(open(sensor_filepath, 'rb'))
                    elif sensor_filepath.split(".")[-1] == 'csv':
                        raw_sensor_data = pd.read_csv(sensor_filepath, index_col=0)
                    if len(raw_sensor_data) > 0:
                        ts_sensor, X_sensor = get_features(raw_sensor_data, sensor)
                    else:
                        ts_sensor, X_sensor = None, None
                    del raw_sensor_data
                featurized_training_data[instance_id][sensor] = (ts_sensor, X_sensor)

        pickle.dump(featurized_training_data, open(featurized_sensor_data_cache, 'wb'))

    raw_sensor_cache = f'{config["cache_dir"]}/sensor_level_data.pb'
    if os.path.exists(raw_sensor_cache):
        X_sensor_data = pickle.load(open(raw_sensor_cache,"rb"))
    else:
        X_sensor_data = dict()
        for sensor in config["sensor_list"]:
            ts_dataset = []
            id_dataset = []
            train_dataset = []
            for instance_id in featurized_training_data.keys():
                if sensor in config['sensor_list']:
                    instance_ts, instance_train = featurized_training_data[instance_id][sensor]
                    if instance_train is None:
                        continue
                    instance_id_tile = np.array([instance_id] * (instance_train.shape[0]))
                    train_dataset.append(instance_train)
                    id_dataset.append(instance_id_tile)
                    ts_dataset.append(instance_ts)
            if len(train_dataset) <= 0:
                continue
            ts_train = np.concatenate(ts_dataset, axis=0)
            id_train = np.concatenate(id_dataset, axis=0)
            X_train = np.concatenate(train_dataset, axis=0)
            df_train = pd.DataFrame(X_train)
            df_train['id'] = id_train
            df_train['ts'] = ts_train.astype(int)
            df_train = df_train.groupby(['id','ts'],as_index=False).max()
            X_sensor_data[sensor] = (df_train['id'], df_train['ts'], df_train.values[:,2:])
        pickle.dump(X_sensor_data, open(raw_sensor_cache, 'wb'))

    featurize_training_data_ts_end = datetime.now()

    logger.info(
        f"Featurized training data in {time_diff(featurize_training_data_ts_start, featurize_training_data_ts_end)} secs.")

    return featurized_training_data, X_sensor_data
