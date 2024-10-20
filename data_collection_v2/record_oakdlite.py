#!/usr/bin/env python3
'''
This is main file to run data collection from depthcam. It collect rgb, and depth information.
Developer: Prasoon Patidar
Created: 5th March, 2022
'''
import datetime

# basic libraries
import queue
import time
import traceback
import numpy as np
import cv2
import depthai as dai
import os
import jstyleson as json
import sys
import signal
import psutil

# Custom libraries
from utils import get_logger
from oakdlite.config import Config
from oakdlite.depth_recorder import DepthRecorderThread
from oakdlite.rgb_recorder import RGBRecorderThread
from oakdlite.pose_recorder import PoseRecorderThread
from oakdlite.poseestimators import get_poseestimator
from oakdlite.run import pose_model_config,create_vpu_pipeline,create_keypoint_frame

CHECKPOINT_FREQ = 60

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

class Config:
    VIDEO_SRC = ''

    # -----video config for oakdlite-----

    # If set (True), the ColorCamera is downscaled from 1080p to 720p.
    # Otherwise (False), the aligned depth is automatically upscaled to 1080p
    downscaleColor = True

    #video recording framerate
    rgb_fps = 10
    pose_fps=10
    depth_fps = 20

    #monocamera resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

    # ----- config for depth/rgb recording -----

    #max duration in single video file
    max_duration_per_video = 30

    #video codec
    video_codec = 'XVID' # for OSX use 'MJPG', for linux 'XVID'


if __name__ == '__main__':

    # initialize logger
    logger = get_logger('oakdlite_runner', 'cache/logs/video/oakdlite')
    logger.info("------------ New Oakdlite Run ------------")
    default_config_file = 'config.json'
    visualize=False
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    t_data_collection_start = datetime.datetime.now()
    start_time = time.time()
    # get experiment dir
    experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    run_config['experiment_dir'] = experiment_dir

    # get VPU pipeline and stereo config
    pipeline, stereo = create_vpu_pipeline()
    logger.info("Created VPU Pipeline")

    # get output queue for rgb info and depth info
    depth_recorder_queue = queue.Queue()
    rgb_recorder_queue = queue.Queue()
    pose_recorder_queue = queue.Queue()

    # initialize depth_recorder thread and rgb_recorder thread
    depth_recorder_thread = DepthRecorderThread(depth_recorder_queue, logger, experiment_dir, Config)
    depth_recorder_thread.start()
    logger.info("Initialized Depth Recorder Thread")

    rgb_recorder_thread = RGBRecorderThread(rgb_recorder_queue, logger, experiment_dir, Config)
    rgb_recorder_thread.start()
    logger.info("Initialized RGB Recorder Thread")

    pose_recorder_thread = PoseRecorderThread(pose_recorder_queue, logger, experiment_dir, Config)
    pose_recorder_thread.start()
    logger.info("Initialized Pose Recorder Thread")

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    if visualize:
        rgbWindowName = "rgb"
        depthWindowName = "depth"
        poseWindowName = 'pose'
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(depthWindowName)
        cv2.namedWindow(poseWindowName)

    # mapping OpenPose keypoints to PoseNet
    pose_estimator = get_poseestimator(pose_model_config, **{"decoder": None})
    num_rgb_frames = 0.
    curr_time_rgb = time.time()
    num_depth_frames = 0.
    curr_time_depth = time.time()
    num_pose_frames = 0.
    curr_time_pose = time.time()
    curr_keypoints = None
    checkpoint = time.time()
    ckpt_file = '/tmp/oakdlite.ckpt'
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
                if time.time() - checkpoint > CHECKPOINT_FREQ:
                    with open(ckpt_file, 'w') as ckpt_f:
                        ckpt_f.write(f'{datetime.datetime.now()}')
                    checkpoint = time.time()
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
                    if visualize:
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
                    if visualize:
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
                    if visualize:
                        keypointFrame = create_keypoint_frame(keypoints, pose_model_config['input_size'][0],
                                                              pose_model_config['input_size'][1], frameRgb.shape[1],
                                                              frameRgb.shape[0],resize_img=False)
                        cv2.imshow(poseWindowName,keypointFrame)
                    if time.time() - curr_time_pose > 10.:
                        logger.info(f"Pose Frames({keypoints.shape}) in 10 secs:{num_pose_frames}")
                        curr_time_pose = time.time()
                        num_pose_frames = 0.
                if visualize:
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                    if cv2.getWindowProperty('rgb', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.getWindowProperty('depth', cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.getWindowProperty('pose', cv2.WND_PROP_VISIBLE) < 1:
                        break
            logger.info(f"Exiting gracefully...")
            pose_recorder_queue.put((None,None))
            depth_recorder_queue.put((None, None))
            rgb_recorder_queue.put((None, None))
    except Exception as e:
        logger.info(f"Exiting with error...")
        logger.info(traceback.print_exc())
        pose_recorder_queue.put((None, None))
        depth_recorder_queue.put((None, None))
        rgb_recorder_queue.put((None, None))
    finally:
        pose_recorder_queue.put((None, None))
        depth_recorder_queue.put((None, None))
        rgb_recorder_queue.put((None, None))
        pose_recorder_thread.stop()
        depth_recorder_thread.stop()
        rgb_recorder_thread.stop()
        pose_recorder_thread.join()
        logger.info("Pose thread joined")
        depth_recorder_thread.join()
        logger.info("Depth thread joined")
        rgb_recorder_thread.join()
        logger.info("RGB thread joined")
        cv2.destroyAllWindows()
        logger.info(f"Oakdlite Sensor closed successfully")