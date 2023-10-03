# Copyright (c) OpenMMLab. All rights reserved.
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

# MODEL_ROOT = 'model_ckpts'
# VIDEO_ROOT = '/mnt/synergy-vax/Phase2/Jan2023_user2'
# OUTPUT_ROOT = '/home/prasoon/video_analysis/model_outputs'
# POSE_CACHE_ROOT = '/home/prasoon/video_analysis/pose_cache_phase2_user2/'
# model_config = {
#     'config': f'{SOURCE_ROOT}/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
#     'checkpoint': f'{SOURCE_ROOT}/checkpoints/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth',
#     'det_config': f'{SOURCE_ROOT}/demo/faster_rcnn_r50_fpn_2x_coco.py',
#     'det_checkpoint': f'{SOURCE_ROOT}/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
#     'label_map': f'{SOURCE_ROOT}/tools/data/skeleton/label_map_ntu120.txt',
#     'pose_config': f'{SOURCE_ROOT}/demo/hrnet_w32_coco_256x192.py',
#     'pose_checkpoint': f'{SOURCE_ROOT}/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
#     'det_score_thr': 0.9,
#     'predict_stepsize': 5,
#     'short_side': 480,
#     'device': 'cuda:1'
# }


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp_skl', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(frame_paths, model, model_config):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    # model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Human Detection')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= model_config['det_score_thr']]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(frame_paths, det_results, model):
    # model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                         args.device)
    ret = []
    print('Pose Estimation')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret
