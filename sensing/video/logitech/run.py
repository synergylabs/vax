#!/usr/bin/env python3
'''
This is main file to run data collection from depthcam. It collect rgb, and depth information.
Developer: Prasoon Patidar
Created: 5th March, 2022
'''

# basic libraries
import queue
import time
import cv2
import pyudev
import os
import traceback
# Custom libraries
from sensing.utils import get_logger
from sensing.video.logitech.recorder import VideoRecorderThread
from sensing.video.logitech.config import Config

def get_device_id():
    context = pyudev.Context()
    devices = pyudev.Enumerator(context)

    path = "/sys/class/video4linux/"
    video_devices = [os.path.join(path, device) for device in os.listdir(path)]
    dev = []
    for device in video_devices:
        udev = pyudev.Devices.from_path(context, device)
        try:
            vid = udev.properties['ID_VENDOR_ID']
            pid = udev.properties['ID_MODEL_ID']
            if vid.lower() == "046d" and pid.lower() == "0825":
                dev.append(int(device.split('video')[-1]))
        except KeyError:
            pass

    # For some reason multiple devices can show up
    if len(dev) > 1:
        for d in dev:
            cam = cv2.VideoCapture(d + cv2.CAP_V4L2)
            data = cam.read()
            cam.release()
            if data[0] == True and data[1] is not None:
                res = d
                break
    elif len(dev) == 1:
        res = dev[0]
    return res

if __name__=='__main__':

    # initialize logger
    logger = get_logger('logitech_video_run','cache/logs/video/logitech')
    logger.info("------------ New Logitech Video Run ------------")


    # get output queue for rgb info and depth info
    video_device_id = get_device_id()
    video_capturer = cv2.VideoCapture(video_device_id)
    if Config.RECORD:
        rgb_recorder_queue = queue.Queue()
        rgb_recorder_thread = VideoRecorderThread(rgb_recorder_queue, time.time_ns(), logger)
        rgb_recorder_thread.start()
        logger.info("Initialized RGB Recorder Thread")
    else:
        logger.warn("RECORDING DISABLED...")
    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "Video"
    cv2.namedWindow(rgbWindowName)

    # start running Video pipeline
    # Connect to device and start pipeline
    try:
        while True:
            frame_time = time.time_ns()
            ret, video_frame = video_capturer.read()
            if ret==True:
                if Config.RECORD:
                    rgb_recorder_queue.put((frame_time, video_frame))
                cv2.imshow(rgbWindowName, video_frame)
            if cv2.waitKey(1) == 27:
                if Config.RECORD:
                    print("Closing Video Recorder")
                    rgb_recorder_thread.stop()
                    rgb_recorder_thread.join()
                # stop2DLidar(serialObj)
                break  # esc to quit
    except Exception as e:
        logger.info(traceback.print_exc())
        rgb_recorder_thread.stop()
        rgb_recorder_thread.join()


