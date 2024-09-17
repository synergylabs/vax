# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
from pathlib import Path
import pandas as pd


# poseNet_skeleton_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9],
#                    [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
#                    [13, 15], [12, 14], [14, 16]]
# def openpose_to_posenet(arr):
#
#     result = np.zeros((17,3), dtype=np.float32)
#     for i,idx in enumerate(openPose_to_poseNet):
#         result[i] = (arr[idx])
#     return result


def posec3d_ntu120_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map

def posec3d_ntu60_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map

def posec3d_hmdb_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_hmdb.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map

def posec3d_ucf_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint-cae8aa4a.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ucf.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map

def stgcn_ntu60_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    stgcn_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return stgcn_model, label_map


def pose_inference(instance_pose_data, posec3d_model, posec3d_label_map, h=480, w=853, short_side=480):

    num_frame = len(instance_pose_data)
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in instance_pose_data])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(instance_pose_data):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(posec3d_model, fake_anno)

    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], xr[1]) for xr in results], columns=['label', 0]).set_index(
        'label')
    return df_scores


def pose_inference_openpose(instance_pose_data, posec3d_model, posec3d_label_map, orig_h=256, orig_w=456, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    # openPose_to_poseNet = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
    num_frame = len(instance_pose_data)
    if num_frame==0:
        return pd.DataFrame(columns=['label', 0]).set_index('label')
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    # num_person = max([len(x) for x in instance_pose_data])
    num_person = 1 #todo: fixed for time being, but need to be dynamic
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, (ts,pose_data) in enumerate(instance_pose_data):
        keypoint[0, i]  = pose_data[0,:,:2]
        keypoint_score[0, i] = pose_data[0,:, 2]
        # for j, pose in enumerate(pose_data):
        #     pose = pose['keypoints']
        #     keypoint[j, i] = pose[openPose_to_poseNet, :2]
        #     keypoint_score[j, i] = pose[openPose_to_poseNet, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    results = inference_recognizer(posec3d_model, fake_anno, return_all_scores=True)
    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], xr[1]) for xr in results], columns=['label', 0]).set_index(
        'label')
    return df_scores
