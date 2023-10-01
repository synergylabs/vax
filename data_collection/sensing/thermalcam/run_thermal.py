#!/usr/bin/env python3
"""
Main driver class for  thermal recording
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
import jstyleson as json
from copy import copy
from flirpy.camera.lepton import Lepton
# custom libraries
from sensing.utils import get_logger, get_screen_size
from sensing.deviceInterface import DeviceInterface
import pickle
import base64
import signal
import psutil
import skimage.measure
# config parameters (not to be changed)
THERMAL_DEVICE_NAME = 'Flir'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 60

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

class FlirReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, logger, viz_queue):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.camera = None
        self.viz_queue = viz_queue

    def start(self):

        # mark this is running
        self.running = True

        # start
        super(FlirReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        try:
            with Lepton() as camera:
                while self.running:
                    img = camera.grab()
                    if img is not None:
                        ts = time.time_ns()
                        self.out_queue.put((ts, img))
                        if self.viz_queue is not None:
                            self.viz_queue.put((ts, img))
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class FlirWriterThread(threading.Thread):
    """
    Writer thread for doppler sensing from awr device
    """

    def __init__(self, in_queue, write_dir, logger,prefix='flir'):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_dir = write_dir
        self.running = False
        self.prefix = prefix
        self.out_file = None
        self.file_start_time = None
        self.checkpoint = time.time()
        self.ckpt_file = '/tmp/thermal.ckpt'

    def start(self):

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {THERMAL_DEVICE_NAME} sensor...")
        # start
        super(FlirWriterThread, self).start()
    def renew_file(self):

        # release older csv
        self.out_file.close()

        # create new csv based on timestamp of next frame and reset current frame number
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.csv', 'w')
        # self.csv_out = csv.writer(self.out_file)
        self.file_start_time = time.time()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        try:
            # connect with device for reading data
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.csv', 'w')
            self.file_start_time = time.time()
            # run till thread is running
            while self.running:
                # run till this video exhausts
                while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
                    if self.running:
                        ts, img_data = self.in_queue.get()
                        img_data = img_data.astype(np.float32)
                        img_data = (img_data / 1e2) - 273
                        # todo: convert image from celcius to kelvin 100k
                        encoded_data = base64.encodebytes(pickle.dumps(img_data)).decode()
                        self.out_file.write(f"{ts} | {encoded_data} ||")
                        if time.time()-self.checkpoint>CHECKPOINT_FREQ:
                            with open(self.ckpt_file,'w') as ckpt_f:
                                ckpt_f.write(f'{datetime.now()}')
                            self.checkpoint = time.time()
                    else:
                        self.out_file.close()
                        break
                if self.running:
                    self.renew_file()
            # with open(self.write_src, 'w') as f:
            #     while self.running:
            #         ts, img_data = self.in_queue.get()
            #         img_data = img_data.astype(np.float32)
            #         img_data = (img_data / 1e2) - 273
            #         # todo: convert image from celcius to kelvin 100k
            #         encoded_data = base64.encodebytes(pickle.dumps(img_data)).decode()
            #         f.write(f"{ts} | {encoded_data} ||")
            #         # f.write(f"{ts} | {str(img_data.tolist())}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class FlirDevice(DeviceInterface):
    """
    Device implementation for FlirDevice
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

        # initialize FlirDevice
        self.name = THERMAL_DEVICE_NAME
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        try:
            with Lepton() as camera:
                img = camera.grab()
            return True
        except:
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = FlirReaderThread(self.sensor_queue,
                                           self.logger, self.viz_queue)
            self.reader.start()

            return True
        except:
            self.logger.error(f"Failed to start reader thread: {traceback.format_exc()}")
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
        try:
            self.write_method = 'csv'
            if self.write_method == 'csv':
                # time_str = datetime.now().strftime("%H%M%S")
                # self.write_src = f"{self.run_config['experiment_dir']}/flir_{time_str}.csv"
                # print(f"Write Source: {self.write_src}")
                self.writer = FlirWriterThread(self.sensor_queue,
                                               self.run_config['experiment_dir'],
                                               self.logger)
                self.writer.start()
                return True
        except:
            self.logger.error(f"Failed to start writer thread, {traceback.format_exc()}")
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
    logger = get_logger("vax_thermal")
    default_config_file = 'config/dc_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'Thermal'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins']*60
    t_data_collection_start = datetime.now()
    start_time = time.time()

    # get experiment dir
    experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
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
    thermalSensor = FlirDevice(run_config, sensor_queue, logger, viz_queue)

    # check if available
    if thermalSensor.is_available():
        logger.info(f"- Found Sensor {thermalSensor.name}-")
        thermalSensor.startReader()
        thermalSensor.startWriter()

    # run for a given max duration
    try:
        # thermal_frames = np.zeros((100, 2))
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, (screen_width) // 2, (4*screen_height) // 5)
        num_frames = 0
        start_viz_time = time.time()
        while (time.time() - start_time < max_duration):
            if visualize:
                if viz_queue.qsize() > 0:
                    # logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                    frame_time, thermal_frame = viz_queue.get()
                    img = thermal_frame.astype(np.float32)
                    img = 255 * (img - img.min()) / (img.max() - img.min())
                    img = skimage.measure.block_reduce(img,(16,16),np.max)
                    # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
                    # You can also try PLASMA or MAGMA
                    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

                    # resizing image for better visualization
                    scale_percent = 8000  # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_name, img_resized)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    num_frames += 1
                if time.time() > start_viz_time + 10.:
                    logger.info(f"Num Thermal Frames in last 10 Secs: {num_frames}")
                    num_frames = 0
                    start_viz_time = time.time()

        thermalSensor.stopWriter()
        thermalSensor.stopReader()
        logger.info(f"Data Collection Complete {thermalSensor.name}")
    except KeyboardInterrupt:
        thermalSensor.stopWriter()
        thermalSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {thermalSensor.name}")
    finally:
        thermalSensor.stopWriter()
        thermalSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {thermalSensor.name}")

