"""
Main driver class for Logitech video recording
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


class LogitechConfig:
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


class LogitechReaderThread(threading.Thread):
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
        super(LogitechReaderThread, self).start()

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


class LogitechVisualizerThread(threading.Thread):
    """
    Visualizer thread for video sensing from logitech device
    """

    def __init__(self, viz_queue, logger, window_name='Video'):
        threading.Thread.__init__(self)
        self.viz_queue = viz_queue
        self.logger = logger
        self.running = False
        self.window_name = window_name

    def start(self):
        self.running = True
        self.logger.info(f"Starting data visualization from {LOGITECH_DEVICE_NAME} sensor...")
        if not super(LogitechVisualizerThread, self).is_alive():
            super(LogitechVisualizerThread, self).start()

    def stop(self):
        # destroy device relevant object
        ...
        # set thread running to False
        self.running = False

    def run(self):
        cv2.namedWindow(self.window_name)
        # self.logger.info(f"Waiting for Running viz data from {YETI_DEVICE_NAME} sensor...")
        while True:
            if self.running:
                if self.viz_queue.qsize() > 0:
                    # self.logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                    frame_time, video_frame = self.viz_queue.get()
                    cv2.imshow(self.window_name, video_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            else:
                time.sleep(2)
        cv2.destroyWindow(self.window_name)


class LogitechWriterThread(threading.Thread):
    def __init__(self, rgb_queue, start_timestamp, logger, write_folder, video_prefix='video'):
        threading.Thread.__init__(self)

        self.in_queue = rgb_queue
        self.max_duration = LogitechConfig.max_duration_per_video
        self.logger = logger

        # if not os.path.exists(write_folder):
        #     os.makedirs(write_folder)

        # initialize output video config
        self.video_width = LogitechConfig.video_width
        self.video_height = LogitechConfig.video_height

        self.video_fps = LogitechConfig.fps
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
        super(LogitechWriterThread, self).start()

    def renew_video(self, timestamp):

        # release older video
        self.video_out.release()

        # create new video based on timestamp of next frame and reset current frame number
        self.video_file = f'{LogitechConfig.data_folder}/{self.video_prefix}_{timestamp}.avi'
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


class LogitechDevice(DeviceInterface):
    """
    Device implementation for Video recording from Logitech
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue=None):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger,viz_queue)

        # initialize LogitechDevice
        self.name = LOGITECH_DEVICE_NAME
        self.device_id = -1
        self.num_channels = -1
        self.sampling_rate = -1
        self.reader = None
        self.writer = None
        self.visualizer=None
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
            self.reader = LogitechReaderThread(self.sensor_queue,
                                               self.device_id,
                                               self.logger,
                                               self.viz_queue)
            self.reader.start()
            if self.viz_queue is not None:
                if self.visualizer is None:
                    self.visualizer = LogitechVisualizerThread(self.viz_queue,
                                                           self.logger)
                self.visualizer.start()
            return True
        else:
            return False

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            if self.viz_queue is not None:
                self.visualizer.stop()
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
            self.writer = LogitechWriterThread(self.sensor_queue,
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
