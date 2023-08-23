"""
Main driver class for  video recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""
from datetime import datetime
# basic libraries
import threading
import queue
import logging
import traceback
import sys
import time
import traceback
import tempfile
from queue import Queue
import sys
import os
import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np
import librosa
import json
import psutil

# custom libraries
from sensing.utils import get_logger, get_screen_size
from sensing.deviceInterface import DeviceInterface

"""
Main driver class for  video recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

# basic libraries
import threading
import queue
import logging
import traceback
import sys
import time
import cv2
import pyudev
import traceback
import tempfile
import queue
import sys
import os
from copy import copy
import sounddevice as sd
import soundfile as sf

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
LOGITECH_DEVICE_NAME = 'Logitech 720p Camera'
LOGITECH_DEVICE_VID = '046d'
LOGITECH_DEVICE_PID = '0825'


class Config:
    # todo: to be removed and merged with main config
    # video recording framerate
    fps = 30
    # video recording frame size
    video_width = 640
    video_height = 480
    # max duration in single video file
    max_duration_per_video = 30
    data_folder = '.'
    # video codec
    video_codec = 'MJPG'  # for OSX use 'MJPG', for linux 'XVID'


class ReaderThread(threading.Thread):
    """
    Reader thread for video sensing from logitech device
    """

    def __init__(self, out_queue, device_id, logger, viz_queue):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.device_id = device_id
        self.video_capturer = None
        self.viz_queue = viz_queue

    def start(self):
        # connect with device for reading data
        self.video_capturer = cv2.VideoCapture(self.device_id)

        # mark this is running
        self.running = True
        self.logger.info(f"Starting reading data from {LOGITECH_DEVICE_NAME} sensor...")
        # start
        super(ReaderThread, self).start()

    def stop(self):
        # destroy device relevant object
        ...
        # set thread running to False
        self.running = False

    def run(self):
        try:
            while self.running:
                frame_time = time.time_ns()
                ret, video_frame = self.video_capturer.read()
                if ret == True:
                    self.out_queue.put((frame_time, video_frame))
                    self.viz_queue.put((copy(frame_time), copy(video_frame)))
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class WriterThread(threading.Thread):
    def __init__(self, rgb_queue, start_timestamp, logger, write_folder, video_prefix='video'):
        threading.Thread.__init__(self)

        self.in_queue = rgb_queue
        self.max_duration = Config.max_duration_per_video
        self.logger = logger

        # if not os.path.exists(write_folder):
        #     os.makedirs(write_folder)

        # initialize output video config
        self.video_width = Config.video_width
        self.video_height = Config.video_height

        self.video_fps = Config.fps
        self.video_prefix = '__'
        self.video_file = f'{write_folder}/{video_prefix}_{start_timestamp}.mkv'
        # self.video_format = cv2.VideoWriter_fourcc(*f'{Config.video_codec}')
        self.video_format = cv2.VideoWriter_fourcc(*'MJPG')
        # output containers
        self.video_out = None
        self.max_frames_per_video = self.max_duration * self.video_fps * 60
        self.current_frame_number = 0
        self.is_running = False

    def start(self):

        # create video output buffer
        self.video_out = cv2.VideoWriter(self.video_file, self.video_format, float(self.video_fps),
                                         (self.video_width, self.video_height))
        self.is_running = True
        super(WriterThread, self).start()

    def renew_video(self, timestamp):

        # release older video
        self.video_out.release()

        # create new video based on timestamp of next frame and reset current frame number
        self.video_file = f'{Config.data_folder}/{self.video_prefix}_{timestamp}.avi'
        self.video_out = cv2.VideoWriter(self.video_file, self.video_format, float(self.video_fps),
                                         (self.video_width, self.video_height))
        self.current_frame_number = 0

    def stop(self):
        # release current video
        self.video_out.release()

        # set thread running to false
        self.is_running = False

    def run(self):

        # run till thread is running
        while self.is_running:
            # run till this video exhausts
            frame_time, frame = self.in_queue.get()
            frame = cv2.rectangle(frame, (0, frame.shape[1]), (frame.shape[0], frame.shape[1] // 2 + 135),
                                  (0, 0, 0), -1)
            frame = cv2.putText(frame, f"{frame_time}",
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255))
            self.video_out.write(frame)
            self.current_frame_number += 1

            if self.current_frame_number >= self.max_frames_per_video:
                self.renew_video(frame_time)


class Device(DeviceInterface):
    """
    Device implementation for Video recording from
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue=None):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

        # initialize Device
        self.name = LOGITECH_DEVICE_NAME
        self.device_id = -1
        self.num_channels = -1
        self.sampling_rate = -1
        self.reader = None
        self.writer = None
        self.visualizer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        self.logger.debug(f"Searching for device name: {self.name}")
        context = pyudev.Context()
        devices = pyudev.Enumerator(context)

        path = "/sys/class/video4linux/"
        video_devices = [os.path.join(path, device) for device in os.listdir(path)]
        matching_devices = []
        for device in video_devices:
            udev = pyudev.Devices.from_path(context, device)
            try:
                vid = udev.properties['ID_VENDOR_ID']
                pid = udev.properties['ID_MODEL_ID']
                if vid.lower() == LOGITECH_DEVICE_VID and pid.lower() == LOGITECH_DEVICE_PID:
                    matching_devices.append(int(device.split('video')[-1]))
            except KeyError:
                pass

        # For some reason multiple devices can show up
        selected_device = -1
        if len(matching_devices) > 1:
            for d in matching_devices:
                cam = cv2.VideoCapture(d + cv2.CAP_V4L2)
                data = cam.read()
                cam.release()
                if data[0] == True and data[1] is not None:
                    selected_device = d
                    break
        elif len(matching_devices) == 1:
            selected_device = matching_devices[0]

        if selected_device > 0:
            self.device_id = selected_device
            return True
        else:
            self.logger.error(f"Device {LOGITECH_DEVICE_NAME} not found..")
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        if self.device_id > 0:
            self.reader = ReaderThread(self.sensor_queue,
                                       self.device_id,
                                       self.logger,
                                       self.viz_queue)
            self.reader.start()

            return True
        else:
            return False

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:

            self.reader.stop()
            self.reader.join()
            return True
        except:
            self.logger.error(f"Failed to stop reader thread, {traceback.format_exc()}")
            return False

    def startWriter(self):
        """
        Start writer thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        if self.device_id > 0:
            self.writer = WriterThread(self.sensor_queue,
                                       time.time_ns(),
                                       self.logger,
                                       self.run_config['experiment_dir'])
            self.writer.start()
            return True
        else:
            return False

    def stopWriter(self):
        """
        Gracefully stop writer thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            self.writer.stop()
            self.writer.join()
            return True
        except:
            self.logger.error(f"Failed to stop writer thread, {traceback.format_exc()}")
            return False


if __name__ == '__main__':
    logger = get_logger("vax_video")
    default_config_file = 'config/dc_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'Video'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins'] * 60
    t_data_collection_start = datetime.now()
    start_time = time.time()
    # Get mount point for harddisk dynamically
    harddisk_uuid = run_config['harddisk_uuid']
    rel_device_path = os.readlink(f'/dev/disk/by-uuid/{harddisk_uuid}')
    device_path_abs = os.path.normpath(os.path.join('/dev/disk/by-uuid',rel_device_path))
    partitions = psutil.disk_partitions()
    disk_mountpath = None 
    for partition in partitions:
        if partition.device==device_path_abs:
            disk_mountpath = partition.mountpoint

    if disk_mountpath is None:
        logger.error(f"Unable to find disk mountpoint for device UUID {harddisk_uuid}")
        sys.exit(1)
    logger.info(f"Found Mount point for disk, {disk_mountpath}")

    # get experiment dir
    experiment_dir = f"{disk_mountpath}/{run_config['name']}/{t_data_collection_start.strftime('%Y%m%d_%H')}"    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    run_config['experiment_dir'] = experiment_dir

    # initialize queues
    sensor_queue = Queue()
    if visualize:
        viz_queue = Queue()
    else:
        viz_queue = None

    # initialize device
    videoSensor = Device(run_config, sensor_queue, logger, viz_queue)

    # check if available
    if videoSensor.is_available():
        logger.info(f"- Found Sensor {videoSensor.name}-")
        videoSensor.startReader()
        videoSensor.startWriter()

    # run for a given max duration
    try:
        video_frames = np.zeros((100, 2))
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, (3*screen_width) // 4, 0)
        num_frames = 0.
        start_viz_time = time.time()
        while (time.time() - start_time < max_duration):
            if visualize:
                if viz_queue.qsize() > 0:
                    # logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                    frame_time, video_frame = viz_queue.get()
                    cv2.imshow(window_name, video_frame)
                    if time.time() > start_viz_time + 10.:
                        logger.info(f"Num Video Frames in last 10 Secs: {num_frames}")
                        num_frames = 0
                        start_viz_time = time.time()
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                    num_frames += 1

    except KeyboardInterrupt:
        videoSensor.stopWriter()
        videoSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {videoSensor.name}")
    finally:
        videoSensor.stopWriter()
        videoSensor.stopReader()
        logger.info(f"Data Collection Complete {videoSensor.name}")
