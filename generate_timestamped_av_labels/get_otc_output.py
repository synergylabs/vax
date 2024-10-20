"""
This file gets raw data from
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
import time
import mmcv
import torch
from mmcv import DictAction
from mmaction.apis import inference_recognizer, init_recognizer

from utils import time_diff
from otc_models import get_model
from otc_models.posec3d import pose_inference
from otc_models.yamnet import audio_inference


def get_otc_output(vax_pipeline_object, config, logger):
    """
    The get_otc_output function takes in a vax_pipeline object and returns the output of the OTC inference.
    The function first checks if there is an existing cache file for the OTC labels, if not it proceeds to run
    the inference on all instances in the vax_pipeline object. The function then returns a dictionary containing
    all of these outputs.

    Args:
        vax_pipeline_object: Get the instances in the vax_pipeline_object
        config: Pass in the configuration file
        logger: Log the messages from the function

    Returns:
        A dictionary of the otc labels

    Doc Author:
        Trelent
    """

    otc_labels = dict()
    otc_labels_ts_start = datetime.now()

    window_length = config["instance_window_length"]
    otc_labels_cache_file = f'{config["cache_dir"]}/otc_labels_{window_length}.pb'

    if os.path.exists(otc_labels_cache_file):
        logger.info("Got OTC labels from cache file...")
        otc_labels = pickle.load(open(otc_labels_cache_file, 'rb'))
    else:
        logger.info("OTC labels cache file not available...")
        otc_instances_cache_dir = f'{config["cache_dir"]}/otc_instances'
        if not os.path.exists(otc_instances_cache_dir):
            os.makedirs(otc_instances_cache_dir)

        # get pose and audio based inferences
        pose_otc_model_names = config['pose_otc_model_names']
        pose_otc_models = {xr: get_model(xr, device='cpu') for xr in pose_otc_model_names}
        # pose_otc_models = {xr: None for xr in pose_otc_model_names}
        pose_otc_labels = dict()

        audio_otc_model_names = config['audio_otc_model_names']
        audio_otc_models = {xr: get_model(xr) for xr in audio_otc_model_names}
        audio_otc_labels = dict()

        for instance_idx, instance_id in enumerate(vax_pipeline_object['instances']):
            # if instance_idx < 3000:
            #     continue
            otc_instance_cache_file = f'{otc_instances_cache_dir}/{instance_id}.pb'
            if os.path.exists(otc_instance_cache_file):
                otc_labels[instance_id] = pickle.load(open(otc_instance_cache_file, 'rb'))
            else:
                instance_pose_data = vax_pipeline_object['instances'][instance_id]['pose_data']
                pose_otc_labels[instance_id] = {}

                if instance_pose_data is not None:
                    for pose_model_name in pose_otc_models:
                        instance_model_cache = f'{config["otc_cache_dir"]}/{pose_model_name}/{instance_id}.mp4.pb'
                        start_time = time.time()
                        if not os.path.exists(instance_model_cache):
                            # df_results = pose_inference(instance_pose_data, pose_otc_models[pose_model_name][0],
                            #                             pose_otc_models[pose_model_name][1])
                            df_results = None
                            # pickle.dump(df_results, open(instance_model_cache,'wb'))
                        else:
                            results_list = pickle.load(open(instance_model_cache,'rb')).split(";")
                            results_list = [xr.split(":") for xr in results_list]
                            results_list = [(label, float(score)) for label, score in results_list]
                            df_results = pd.DataFrame(results_list, columns=['label', 0]).set_index('label')
                        pose_otc_labels[instance_id][pose_model_name] = df_results
                        logger.info(f"Got pose inference for {instance_id}, {pose_model_name} in {time.time()-start_time:.3f} secs..")

                instance_audio_data = vax_pipeline_object['instances'][instance_id]['audio_data']
                audio_otc_labels[instance_id] = {}
                if instance_audio_data is not None:
                    for audio_model_name in audio_otc_models:
                        instance_model_cache = f'{config["otc_cache_dir"]}/{audio_model_name}/{instance_id}.mp4.pb'
                        start_time = time.time()
                        if not os.path.exists(instance_model_cache):
                            if len(pose_otc_labels[instance_id].keys()) <= 2:
                                continue
                            df_results = audio_inference(instance_audio_data, audio_otc_models[audio_model_name][0],
                                                     audio_otc_models[audio_model_name][1])
                            pickle.dump(df_results, open(instance_model_cache,'wb'))
                        else:
                            df_results = pd.read_csv(instance_model_cache,index_col=0)
                        logger.info(
                            f"Got audio inference for {instance_id}, {audio_model_name} in {time.time() - start_time:.3f} secs..")
                        audio_otc_labels[instance_id][audio_model_name] = df_results

                otc_labels[instance_id] = deepcopy(pose_otc_labels[instance_id])
                otc_labels[instance_id].update(audio_otc_labels[instance_id])
                pickle.dump(otc_labels[instance_id], open(otc_instance_cache_file, 'wb'))
                logger.info(f"Got otc labels for new instance(I-{instance_idx}):{instance_id}...")
        pickle.dump(otc_labels, open(otc_labels_cache_file, 'wb'))

    otc_labels_ts_end = datetime.now()

    logger.info(
        f"Got off the shelf av model labels in {time_diff(otc_labels_ts_start, otc_labels_ts_end)}")

    return otc_labels
