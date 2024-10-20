import glob
import logging
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
import soundfile as sf

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture

from datetime import datetime
import moviepy.editor as mp

from utils import time_diff
import argparse
import os
import os.path as osp
import shutil
import time
from datetime import datetime
import glob
import traceback
import sys
import pickle

import cv2
import mmcv
import numpy as np
import torch
from otc_models.pose_extraction import init_pose_model, init_detector, frame_extraction, detection_inference, \
    pose_inference
from mmcv import DictAction

from mmaction.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

MODEL_CHECKPOINT_DIR = 'otc_models/model_ckpts'
MODEL_CONFIG_DIR = 'otc_models/model_configs'
POSE_CACHE_ROOT = '/Users/ppatida2/VAX/vax-public/cache/pose_cache'
AUDIO_CACHE_ROOT = '/Users/ppatida2/VAX/vax-public/cache/audio_cache'
os.makedirs(POSE_CACHE_ROOT, exist_ok=True)

pose_extraction_config = {
    'config': f'{MODEL_CONFIG_DIR}/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
    'det_config': f'{MODEL_CONFIG_DIR}/faster_rcnn_r50_fpn_2x_coco.py',
    'det_checkpoint': f'{MODEL_CHECKPOINT_DIR}/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
    'pose_config': f'{MODEL_CONFIG_DIR}/hrnet_w32_coco_256x192.py',
    'pose_checkpoint': f'{MODEL_CHECKPOINT_DIR}/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    'det_score_thr': 0.9,
    'predict_stepsize': 5,
    'short_side': 480,
    'device': 'cpu'
}

def get_instance_raw_av_data(config, logger):
    vax_pipeline_object_cache = f'{config["cache_dir"]}/vax_pipeline_object.pb'
    if os.path.exists(vax_pipeline_object_cache):
        logger.info("Got vax_pipeline_object from cache file...")
        vax_pipeline_object = pickle.load(open(vax_pipeline_object_cache, 'rb'))
    else:
        vax_pipeline_object = dict()
        processed_data_dir = config["data_dir"]
        pose_cache_data_dir = config["pose_cache_dir"]
        vax_pipeline_object['instances'] = dict()
        video_files = glob.glob(f"{processed_data_dir}/*.mp4")
        for video_file in video_files:
            groundtruth_activity = "Unknown"
            instance_id = video_file.split("/")[-1].split(".")[0]
            # check if a/v raw data exists
            instance_audio_file = video_file.replace(".mp4", ".wav")
            instance_video_file = video_file
            instance_pose_cache_file = pose_cache_data_dir + f"/{instance_id}.mp4.pb"
            vax_pipeline_object['instances'][instance_id] = {
                'activity_label': groundtruth_activity,
                'instance_id': instance_id,
                'instance_video_file': instance_video_file,
                'instance_audio_file': instance_audio_file,
                'instance_pose_file': instance_pose_cache_file
            }
        # get pose and audio data
        vax_pipeline_object['instances'] = get_instance_audio_info(vax_pipeline_object['instances'])
        vax_pipeline_object['instances'] = get_instance_pose_info(vax_pipeline_object['instances'])
        pickle.dump(vax_pipeline_object, open(vax_pipeline_object_cache, 'wb'))

    return vax_pipeline_object


def get_instance_audio_info(all_instances):
    for instance_id in all_instances:
        instance_audio_file = all_instances[instance_id]['instance_audio_file']
        audio_data, audio_sr = sf.read(instance_audio_file, dtype=np.int16)
        all_instances[instance_id].update({
            'audio_data': (audio_data, audio_sr)
        })

    return all_instances


def get_instance_pose_info(all_instances):
    # init detector model
    model_det = init_detector(pose_extraction_config['det_config'], pose_extraction_config['det_checkpoint'],
                              pose_extraction_config['device'])
    assert model_det.CLASSES[0] == 'person', ('We require you to use a detector '
                                              'trained on COCO')

    # init pose model
    model_pose = init_pose_model(pose_extraction_config['pose_config'], pose_extraction_config['pose_checkpoint'],
                                 pose_extraction_config['device'])
    model_config = mmcv.Config.fromfile(pose_extraction_config['config'])

    for instance_id in all_instances:
        pose_cache_file = all_instances[instance_id]['instance_pose_file']
        pose_results = None
        if os.path.exists(pose_cache_file):
            pose_results = pickle.load(open(pose_cache_file, 'rb'))
            print("Got pose results from cache..")
        else:
            "No pose results for this instance, continuing.."
        #
        #     continue
        #     print(f"Getting pose results for {instance_av_file}..")
        #     frame_paths, original_frames = frame_extraction(instance_av_file,
        #                                                     pose_extraction_config['short_side'])
        #     h, w, _ = original_frames[0].shape
        #
        #     # Get clip_len, frame_interval and calculate center index of each clip
        #
        #     for component in model_config.data.test.pipeline:
        #         if component['type'] == 'PoseNormalize':
        #             component['mean'] = (w // 2, h // 2, .5)
        #             component['max_value'] = (w, h, 1.)
        #
        #     # Get Human detection results
        #     det_results = detection_inference(frame_paths, model_det, pose_extraction_config)
        #     torch.cuda.empty_cache()
        #     pose_results = pose_inference(frame_paths, det_results, model_pose)
        #     torch.cuda.empty_cache()
        #     pickle.dump(pose_results, open(pose_cache_file, 'wb'))
        #     tmp_frame_dir = osp.dirname(frame_paths[0])
        #     shutil.rmtree(tmp_frame_dir)

        all_instances[instance_id].update({
            'pose_data': pose_results
        })
    return all_instances

# get_instance_raw_av_data('/Volumes/Vax Storage/processed_data/P7', logging.getLogger("Logs"))
