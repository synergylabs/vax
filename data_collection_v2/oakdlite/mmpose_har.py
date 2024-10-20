#!/usr/bin/env python3
'''
This is main file to run data collection from depthcam. It collect rgb, and depth information.
Developer: Prasoon Patidar
Created: 5th March, 2022
'''

# basic libraries
import queue
import time
import traceback
import numpy as np
import cv2
import depthai as dai
import os
import json

# Custom libraries
from sensing.utils import get_logger
from sensing.video.oakdlite.config import Config
from sensing.video.oakdlite.depth_recorder import DepthRecorderThread
from sensing.video.oakdlite.rgb_recorder import RGBRecorderThread
from sensing.video.oakdlite.pose_recorder import PoseRecorderThread
from sensing.video.oakdlite.poseestimators import get_poseestimator

MODELS_FOLDER = "/home/vax/vax-codebase/cache/oakdlite_models"
MODEL_NAME = 'openpose1'
VISUALIZE=False
'''
@ref: Kevin Schlegel, Oak-HumanPoseEstimation
'''


def get_model_list():
    """
    Reads the configurations of all available models from file.

    The 'models.json' file contains the config dictionaries for all included
    models.

    Returns
    -------
    model_list : dict
        Dictionary of all model configurations
    """
    with open(os.path.join(MODELS_FOLDER, "models.json"), "r") as model_file:
        model_list = json.load(model_file)
    return model_list


model_config = get_model_list()[MODEL_NAME]

'''
This function creates pipeline to run on oakdlite edge processor for collecting and processing data
'''


def create_vpu_pipeline():
    # Create pipeline
    pipeline = dai.Pipeline()
    if not MODEL_NAME=='openpose1':
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    pose = pipeline.createNeuralNetwork()

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(Config.fps)
    if model_config["color_order"] == "BGR":
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    elif model_config["color_order"] == "RGB":
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    else:
        raise ValueError("Invalid color order: {}".format(
            model_config["color_order"]))
    camRgb.setPreviewSize(*model_config["input_size"])
    camRgb.setInterleaved(False)
    if Config.downscaleColor: camRgb.setIspScale(2, 3)
    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    camRgb.initialControl.setManualFocus(130)

    left.setResolution(Config.monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(Config.fps)
    right.setResolution(Config.monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(Config.fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.disparity.link(disparityOut.input)

    blob = (model_config["blob"] + "_sh6.blob")
    path = os.path.join(MODELS_FOLDER, blob)
    pose.setBlobPath(path)
    camRgb.preview.link(pose.input)

    pose_out = pipeline.createXLinkOut()
    pose_out.setStreamName("pose")
    queueNames.append("pose")
    pose.out.link(pose_out.input)

    return pipeline, stereo


def create_keypoint_frame(keypoints, src_frame_h, src_frame_w, target_frame_h, target_frame_w,point_radius=5,resize_img=True):
    keypoint_img = np.zeros((src_frame_w, src_frame_h, 3), dtype=np.uint8)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for person_idx in range(keypoints.shape[0]):
        person_kps = keypoints[person_idx]
        for (x, y, p) in person_kps:
            if p > 0.3:
                keypoint_img = cv2.circle(keypoint_img, (int(x), int(y)), point_radius, color[person_idx % len(color)], -1)
    if resize_img:
        keypoint_img = cv2.resize(keypoint_img, (target_frame_h, target_frame_w), interpolation=cv2.INTER_AREA)
    return keypoint_img


if __name__ == '__main__':

    # initialize logger
    logger = get_logger('oakdlite_runner', 'cache/logs/video/oakdlite')
    logger.info("------------ New DepthCam Run ------------")

    # get VPU pipeline and stereo config
    pipeline, stereo = create_vpu_pipeline()
    logger.info("Created VPU Pipeline")

    # get output queue for rgb info and depth info
    depth_recorder_queue = queue.Queue()
    rgb_recorder_queue = queue.Queue()
    pose_recorder_queue = queue.Queue()

    # initialize depth_recorder thread and rgb_recorder thread
    depth_recorder_thread = DepthRecorderThread(depth_recorder_queue, time.time_ns(), logger)
    depth_recorder_thread.start()
    logger.info("Initialized Depth Recorder Thread")

    rgb_recorder_thread = RGBRecorderThread(rgb_recorder_queue, time.time_ns(), logger)
    rgb_recorder_thread.start()
    logger.info("Initialized RGB Recorder Thread")

    pose_recorder_thread = PoseRecorderThread(pose_recorder_queue, time.time_ns(), logger)
    pose_recorder_thread.start()
    logger.info("Initialized Pose Recorder Thread")

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    if VISUALIZE:
        rgbWindowName = "rgb"
        depthWindowName = "depth"
        poseWindowName = 'pose'
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(depthWindowName)
        cv2.namedWindow(poseWindowName)

    # mapping OpenPose keypoints to PoseNet
    pose_estimator = get_poseestimator(model_config, **{"decoder": None})
    num_rgb_frames = 0.
    curr_time_rgb = time.time()
    num_depth_frames = 0.
    curr_time_depth = time.time()
    num_pose_frames = 0.
    curr_time_pose = time.time()
    curr_keypoints = None
    # start running VPU pipeline
    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:
            logger.info("Oakdlite device found, starting processing")
            frameRgb = None
            frameDisp = None

            while True:
                latestPacket = {}
                latestPacket["rgb"] = None
                latestPacket["disp"] = None
                latestPacket["pose"] = None

                queueEvents = device.getQueueEvents(("rgb", "disp", "pose"))
                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        # logger.info(f"Queue packets: {len(packets)}")
                        latestPacket[queueName] = packets[-1]

                frame_time = time.time_ns()
                if latestPacket["rgb"] is not None:
                    # send rgb frame
                    frameRgb = latestPacket["rgb"].getCvFrame()
                    rgb_recorder_queue.put((frame_time, frameRgb))
                    num_rgb_frames += 1
                    if time.time() - curr_time_rgb > 10.:
                        logger.info(f"RGB Frames({frameRgb.shape}) in 10 secs:{num_rgb_frames}")
                        curr_time_rgb = time.time()
                        num_rgb_frames = 0.

                    if curr_keypoints is not None:
                        keypointFrame = create_keypoint_frame(curr_keypoints, model_config['input_size'][0],
                                                              model_config['input_size'][1], frameRgb.shape[1],
                                                              frameRgb.shape[0])
                        frameRgb = cv2.add(frameRgb,keypointFrame)
                    if VISUALIZE:
                        cv2.imshow(rgbWindowName, frameRgb)

                if latestPacket["disp"] is not None:
                    # logger.info("Put Object in queue")
                    frameDisp = latestPacket["disp"].getFrame()
                    maxDisparity = stereo.initialConfig.getMaxDisparity()
                    # Optional, extend range 0..95 -> 0..255, for a better visualisation
                    if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
                    # Optional, apply false colorization
                    if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
                    frameDisp = np.ascontiguousarray(frameDisp)
                    if VISUALIZE:
                        cv2.imshow(depthWindowName, frameDisp)
                    depth_recorder_queue.put((frame_time, frameDisp))
                    num_depth_frames += 1
                    if time.time() - curr_time_depth > 10.:
                        logger.info(f"Depth Frames({frameDisp.shape}) in 10 secs:{num_depth_frames}")
                        curr_time_depth = time.time()
                        num_depth_frames = 0.

                if latestPacket["pose"] is not None:
                    nn_out = latestPacket["pose"]
                    keypoints = pose_estimator.get_pose_data(nn_out)
                    # curr_keypoints = keypoints
                    pose_recorder_queue.put((frame_time, keypoints))
                    num_pose_frames += 1
                    keypointFrame = create_keypoint_frame(keypoints, model_config['input_size'][0],
                                                          model_config['input_size'][1], frameRgb.shape[1],
                                                          frameRgb.shape[0],resize_img=False)
                    if VISUALIZE:
                        cv2.imshow(poseWindowName,keypointFrame)
                    if time.time() - curr_time_pose > 10.:
                        logger.info(f"Pose Frames({keypoints.shape}) in 10 secs:{num_pose_frames}")
                        curr_time_pose = time.time()
                        num_pose_frames = 0.
                if VISUALIZE:
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit

    except Exception as e:
        logger.info(f"Exiting gracefully...")
        logger.info(traceback.print_exc())
        depth_recorder_thread.stop()
        rgb_recorder_thread.stop()
        pose_recorder_thread.stop()
        depth_recorder_thread.join()
        rgb_recorder_thread.join()
        pose_recorder_thread.join()
        logger.info(f"Sensor closed successfully")
