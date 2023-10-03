"""
This is main function that takes input as raw data from a structured directory and do end to end training and evaluation
"""
import glob
import sys
import traceback

import pandas as pd
import numpy as np
from queue import Queue
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import json
import time
from collections import Counter
import os
import pickle
from copy import deepcopy
import itertools
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import *
from cleanlab.classification import CleanLearning


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def merge_dicts(dicts):
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union == 0:
        return 0.
    return float(intersection) / union


from trainer import get_clf
from featurize import get_features
from ensemble import get_ensemble
from . import context_activities, activity_context_map, instance_filters
from utils import get_logger


def merge_dicts(dicts):
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)


def train_v2_models(user_id, logger, data_dir, df_av_labels, train_config):
    logger.info(f"------------------------Training PVS models for User: {user_id}------------------------")
    user_data_dir = f'{data_dir}/{user_id}'
    feature_cache_file = f'{train_config["feature_cache_dir"]}/{user_id}_{train_config["featurizer"]}.pb'

    if os.path.exists(feature_cache_file):
        featurized_instance_data = pickle.load(open(feature_cache_file, "rb"))
    else:
        # featurize data all instances
        featurized_instance_data = {}
        activity_dirs = glob.glob(f'{user_data_dir}/*')
        for activity_dir in activity_dirs:
            instance_dirs = glob.glob(f'{activity_dir}/*')
            for instance_dir in instance_dirs:
                logger.info(instance_dir)
                instance_id = instance_dir.split("/")[-1]
                if instance_id in instance_filters:
                    continue
                if 'BAD' in instance_dir:
                    continue

                featurized_instance_data[instance_id] = {'id': instance_id}

                for sensor_idx, sensor in enumerate(train_config['sensor_list']):
                    sensor_filepath = f"{instance_dir}/{train_config['sensor_files'][sensor_idx]}"
                    ts_sensor, X_sensor = None, None
                    if os.path.exists(sensor_filepath):
                        if sensor_filepath.split(".")[-1] == 'pb':
                            raw_sensor_data = pickle.load(open(sensor_filepath, 'rb'))
                        elif sensor_filepath.split(".")[-1] == 'csv':
                            raw_sensor_data = pd.read_csv(sensor_filepath, index_col=0)
                        if len(raw_sensor_data) > 0:
                            ts_sensor, X_sensor = get_features(raw_sensor_data, sensor, train_config['featurizer'])
                        else:
                            ts_sensor, X_sensor = None, None
                        del raw_sensor_data
                    featurized_instance_data[instance_id][sensor] = (ts_sensor, X_sensor)
        # store featurized data into cache
        pickle.dump(featurized_instance_data, open(feature_cache_file, "wb"))

    av_files_id = train_config['av_labels_file'].split("/")[-1].split(".csv")[0]
    results_file = f'{train_config["results_cache_dir"]}/{user_id}_{av_files_id}_{"-".join(train_config["trainer"])}.csv'
    if os.path.exists(results_file):
        logger.info(f"Results file {results_file} already present, skipping user {user_id}")

    sensor_data_dict = {}
    for sensor in train_config['sensor_list']:
        id_sensor, ts_sensor, X_sensor, gt_sensor, pred_sensor = [], [], [], [], []
        for instance_id in featurized_instance_data.keys():
            instance_ts, instance_X = featurized_instance_data[instance_id][sensor]
            if instance_id in df_av_labels.instance.unique():
                instance_gt = df_av_labels[df_av_labels.instance == instance_id]['gt'].values[0]
                instance_pred = df_av_labels[df_av_labels.instance == instance_id]['prediction'].values[0]
                if instance_ts is not None:
                    ts_sensor.append(instance_ts)
                    X_sensor.append(instance_X)
                    gt_sensor.append([instance_gt] * instance_X.shape[0])
                    pred_sensor.append([instance_pred] * instance_X.shape[0])
                    id_sensor.append([instance_id] * instance_X.shape[0])
        if len(id_sensor) > 0.:
            id_sensor = np.concatenate(id_sensor)
            ts_sensor = np.concatenate(ts_sensor)
            X_sensor = np.concatenate(X_sensor)
            gt_sensor = np.concatenate(gt_sensor)
            pred_sensor = np.concatenate(pred_sensor)
            sensor_data_dict[sensor] = (id_sensor, ts_sensor, X_sensor, gt_sensor, pred_sensor)
        else:
            sensor_data_dict[sensor] = (None, None, None, None, None)

    instance_ids = df_av_labels['instance'].values
    groundtruth_marked_ids = df_av_labels[df_av_labels.score==-1].instance.unique().tolist()
    logger.info(f"Groundtruth marked ids: {groundtruth_marked_ids}")
    groundtruth = df_av_labels['gt'].values
    instance_av_label = df_av_labels['prediction'].values

    results_dict = {}
    privacy_clfs = {xr: get_clf(xr) for xr in train_config['trainer']}
    user_training_cache_dir = f'{train_config["training_cache_dir"]}/{results_file.split("/")[-1].split(".csv")[0]}/'
    if not os.path.exists(user_training_cache_dir):
        os.makedirs(user_training_cache_dir)

    instance_cache_dir = f'{train_config["training_cache_dir"]}/instances/{av_files_id}/{user_id}'
    if not os.path.exists(instance_cache_dir):
        os.makedirs(instance_cache_dir)

    for sensor in train_config['sensor_list']:
        sensor_results = []
        logger.info(f"Evaluating for sensor {sensor}")
        id_sensor, ts_sensor, X_sensor, gt_sensor, av_sensor = sensor_data_dict[sensor]
        if id_sensor is None:
            logger.info(f"No Data for sensor {sensor} for user {user_id}. Skipping...")
            continue
        sensor_cache = f'{user_training_cache_dir}/{sensor}.pb'
        if os.path.exists(sensor_cache):
            results_dict[sensor] = pickle.load(open(sensor_cache, 'rb'))
        else:
            for instance_idx, test_instance in enumerate(instance_ids):
                if test_instance in groundtruth_marked_ids:
                    logger.info(f"Skipping GT marked instance: {test_instance}")
                    continue
                test_idxes = np.where(id_sensor == test_instance)[0]
                test_context = activity_context_map[groundtruth[instance_idx]]
                if len(test_idxes) > 0.:
                    # logger.info(
                    #     f"Prediction for instance:{test_instance}, GT: {groundtruth[instance_idx]}, AV:{instance_av_label[instance_idx]}")
                    start_time = time.time()
                    X_test, y_test = X_sensor[test_idxes, :], av_sensor[test_idxes]
                    train_idxes = \
                        np.where(
                            (~(id_sensor == test_instance)) & (
                                np.isin(gt_sensor, context_activities[test_context])))[
                            0]
                    X_train, y_train = X_sensor[train_idxes, :], av_sensor[train_idxes]

                    # filter undetected values
                    detected_idx = np.where(~(y_train == 'Undetected'))[0]
                    X_train, y_train = X_train[detected_idx], y_train[detected_idx]


                    # Balance class distribution using SMOTE
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y_train)
                    _, counts = np.unique(y_encoded, return_counts=True)
                    try:
                        smote_model = SMOTE(sampling_strategy='not majority', k_neighbors=min(counts) - 1)
                        X_sm, y_sm = smote_model.fit_resample(X_train, y_encoded)
                    except:
                        logger.info(f"{sensor}_{test_instance}-Error in minority oversampling")
                        logger.info(traceback.format_exc())
                        continue
                    
                    logger.info(f"{test_instance}: Data Preparation({time.time()-start_time:.3f} secs): {X_sm.shape}")
                    for pTrainer in privacy_clfs.keys():
                        instance_trainer_cache = f'{instance_cache_dir}/{test_instance}_{sensor}_{pTrainer}.pb'
                        pred_dict = None
                        start_time = time.time()
                        if os.path.exists(instance_trainer_cache):
                            pred_dict = pickle.load(open(instance_trainer_cache, 'rb'))
                        else:
                            try:
                                # get base classifier from trainer
                                base_clf_ = privacy_clfs[pTrainer]()
                                # build classifier with clean learning
                                clean_clf = CleanLearning(clf=base_clf_)
                                # fit and predict with clean learning wrapper
                                clean_clf.fit(X_sm, y_sm)
                                y_pred_encoded = clean_clf.predict(X_test)
                                y_pred_proba = clean_clf.predict_proba(X_test).max(axis=1)
                                y_pred = label_encoder.inverse_transform(y_pred_encoded)
                                df_pred = \
                                    pd.DataFrame(zip(y_pred, y_pred_proba),
                                                 columns=['prediction', 'score']).groupby(
                                        'prediction')[
                                        'score'].sum()
                                pred_dict = (df_pred / df_pred.sum()).to_dict()
                                pickle.dump(pred_dict,open(instance_trainer_cache, 'wb'))
                            except KeyboardInterrupt:
                                sys.exit(0)
                            except:
                                logger.info(f"Error in instance:{instance_id},trainer:{pTrainer}")
                                logger.info(traceback.format_exc())
                        if pred_dict is not None:
                            logger.info(f"{test_instance}-{groundtruth[instance_idx]}-{instance_av_label[instance_idx]}-{pTrainer}({time.time() - start_time:.3f} secs):{pred_dict}")
                            sensor_results.append(
                                (test_instance, groundtruth[instance_idx], instance_av_label[instance_idx], pTrainer,
                                pred_dict))
            df_sensor_results = pd.DataFrame(sensor_results,
                                             columns=['instance_id', 'gt_label', 'av_label', 'trainer',
                                                      'prediction_dict'])
            pickle.dump(df_sensor_results, open(sensor_cache, 'wb'))
            results_dict[sensor] = df_sensor_results.copy(deep=True)

    df_final_all = None
    for sensor in results_dict.keys():
        df_sensor = results_dict[sensor]
        df_sensor['sensor'] = sensor
        if df_final_all is None:
            df_final_all = df_sensor.copy(deep=True)
        else:
            df_final_all = pd.concat([df_final_all, df_sensor])

    df_final_all['sensor_label'] = df_final_all['prediction_dict'].apply(lambda x: max(x, key=x.get))
    df_final_all.to_csv(results_file)
    logger.info(f"------------------------Training complete for User: {user_id}------------------------")
    return None


if __name__ == '__main__':
    try:
        users = sys.argv[1].split(",")
        print(f"Got Users for training: {users}")
    except:
        print("user not provided, using all users")
        users = [f'P{i}' for i in range(1,11)]
    for user in users:
        train_config = {
            'av_labels_file': '/home/prasoon/vax/cache/av_optics_results/avx_0.75_0.9_new_1sample.csv',
            'featurizer': 'm2',
            'trainer': ['svmClean','knnClean'],
            'feature_cache_dir': f'/home/prasoon/vax/cache/privacy_models/features',
            'n_cv_splits': 5,
            'sensor_list': ['doppler', 'lidar', 'thermal', 'micarray', 'ENV', 'PIR'],
            'sensor_files': ['doppler.pb', 'lidar2d.pb', 'thermal.pb', 'micarray.pb', 'mites.csv', 'mites.csv'],
            'results_cache_dir': f'/home/prasoon/vax/cache/privacy_models/results_v10',
            'training_cache_dir': f'/home/prasoon/vax/cache/privacy_models/training_v10',
            'base_data_dir': '/Volumes/Vax Storage/processed_data'
        }

        # get av labels
        av_files_id = train_config['av_labels_file'].split("/")[-1].split(".csv")[0]
        df_av_labels = pd.read_csv(train_config['av_labels_file'])
        df_av_labels = df_av_labels[['user', 'instance_id', 'groundtruth', 'best_prediction','best_score']]
        df_av_labels.columns = ['user', 'instance', 'gt', 'prediction','score']
        df_av_labels = df_av_labels.sort_values(by=['user','gt'])
        data_dir = train_config['base_data_dir']

        if not os.path.exists(train_config['feature_cache_dir']):
            os.makedirs(train_config['feature_cache_dir'])
        if not os.path.exists(train_config['results_cache_dir']):
            os.makedirs(train_config['results_cache_dir'])

        logger = get_logger(f'{train_config["featurizer"]}_{"-".join(train_config["trainer"])}_{av_files_id}')
        train_v2_models(user, logger, data_dir, df_av_labels, train_config)
