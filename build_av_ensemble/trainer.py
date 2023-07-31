"""
This file trains audio video ensemble from reference homes.
"""
import datetime
import os.path
import traceback
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from HAR.av.AVensemble.config import OpticsConfig
import glob
import pickle
from copy import deepcopy
from sklearn.cluster import OPTICS
import warnings
import random
import sys

warnings.filterwarnings("ignore")

from HAR.av.AVensembleNNPhase1.util import merge_dicts, jaccard_score_custom, aggregate_audio_ts_scores, fetch_model_data
from HAR.av.AVensembleNNPhase1.audio_model import train_audio_ensemble, predict_from_audio_ensemble
from HAR.av.AVensembleNNPhase1.video_model import train_video_ensemble, predict_from_video_ensemble
from HAR.av.AVensembleNNPhase1.clf_model import train_clf_ensemble, predict_from_clf_ensemble
import xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def train_av_model(train_raw_data,
                   audio_model_list=('yamnet',),
                   video_model_list=('yamnet', 'posec3d_hmdb', 'posec3d_ntu60', 'posec3d_ntu120', 'posec3d_ucf','stgcn_ntu60')
                   ):

    parameter_grid_mlp2 = {
        'hidden_layer_sizes': [(100, 50), (50, 20), (20, 10)],
        'activation': ['relu']
    }
    # train video ensemble
    video_ensemble = train_clf_ensemble(train_raw_data,
                                        clf_=MLPClassifier(),
                                        gridsearch_param_grid=parameter_grid_mlp2,
                                        model_list=video_model_list)

    # train audio ensemble
    audio_ensemble = train_clf_ensemble(train_raw_data,
                                        clf_=MLPClassifier(),
                                        gridsearch_param_grid=parameter_grid_mlp2,
                                        model_list=audio_model_list)

    return video_ensemble, audio_ensemble


def predict_from_av_model(test_raw_data, video_ensemble, audio_ensemble):
    # prediction from video ensemble
    video_predictions, video_predictions_detailed = predict_from_clf_ensemble(test_raw_data, video_ensemble)

    # predictions from audio ensemble
    audio_predictions, audio_predictions_detailed = predict_from_clf_ensemble(test_raw_data, audio_ensemble)

    # combined output from audio and video ensembles
    final_predictions = combine_av_predictions(audio_predictions, audio_predictions_detailed,
                                               video_predictions, video_predictions_detailed)

    predictions_dict = {
        'audio': audio_predictions,
        'video': video_predictions,
        'details': {
            'audio': audio_predictions_detailed,
            'video': video_predictions_detailed
        },
        'combined': final_predictions
    }
    return predictions_dict


def combine_av_predictions(audio_predictions, audio_predictions_detailed,
                           video_predictions, video_predictions_detailed):
    return audio_predictions


if __name__ == '__main__':
    # test 1: run a single set of train and test users
    # curr_time = datetime.datetime.now().strftime("%Y%m%d_%H_nn_verif2")
    # curr_time = '20230215_01'
    model_list = ('yamnet', 'posec3d_hmdb', 'posec3d_ntu60', 'posec3d_ntu120', 'posec3d_ucf',
                  'stgcn_ntu60')
    users_all = [f'P{i}' for i in range(1,11)]
    # max_combinations = len(users_all)
    # num_test_users = int(sys.argv[1])
    num_test_users = 1
    print(f"Creating ensemble for test combination for {num_test_users} users")

    test_user_sets = list(map(tuple,combinations(users_all, num_test_users)))
    # test_user_set_idxs = np.random.choice(len(test_user_sets),size=max_combinations,replace=False)
    # test_user_sets = [test_user_sets[idx] for idx in test_user_set_idxs]
    print("test_user_sets",test_user_sets)
    ROOT_DIR = f'/Users/ppatida2/VAX'
    if not os.path.exists(ROOT_DIR):
        ROOT_DIR = f'/home/prasoon'

    # cache_dir = f'{ROOT_DIR}/vax/cache/av_results_optics/results_nn_diff_homes/test_count_{num_test_users}'
    cache_dir = f'{ROOT_DIR}/vax/cache/av_results_optics/results_nn_all_homes_pandas_'
    os.makedirs(cache_dir,exist_ok=True)
    for test_users in test_user_sets[:1]:
        print(f"Test Users: {test_users}")
        cache_file = f'{cache_dir}/{"_".join(sorted(test_users))}.pb'
        try:
            if not os.path.exists(cache_file):
                # train_users = [usr for usr in users_all if (usr not in test_users)]
                train_users = users_all
                train_raw_data, test_raw_data = fetch_model_data(model_list, train_users, test_users)
                video_ensemble, audio_ensemble = train_av_model(train_raw_data)
                predictions_dict = predict_from_av_model(test_raw_data, video_ensemble, audio_ensemble)
                pickle.dump((predictions_dict, video_ensemble, audio_ensemble), open(cache_file, 'wb'))
                print(f"Created ensemble for: {sorted(train_users)}, with predictions on {sorted(list(test_users))}")
            else:
                print(f"Ensemble exists for: {sorted(train_users)}, with predictions on {sorted(list(test_users))}")
        except:
            print(f"Ensemble creation error: {sorted(train_users)}, with predictions on {sorted(list(test_users))}")
            print(traceback.format_exc(()))