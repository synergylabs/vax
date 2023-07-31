"""
Main driver class for doppler data collection from awr1642boost-ods
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

# basic libraries
import threading
import time
import queue
import logging
import traceback
import sys
import numpy as np
import cv2
from flirpy.camera.lepton import Lepton
import serial
import os
import socket
import pickle
from pathlib import Path
import tempfile
from serial.tools import list_ports
import nmap
from dateutil import parser

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
THERMAL_DEVICE_NAME = 'Flir'


class FlirReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.camera = None

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

    def __init__(self, in_queue, write_src, logger):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_src = write_src
        self.is_running = False

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {THERMAL_DEVICE_NAME} sensor...")
        # start
        super(FlirWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        is_header_set = False
        try:
            with open(self.write_src, 'w') as f:
                while self.running:
                    if not is_header_set:
                        f.write("ts | data\n")
                    ts, img_data = self.in_queue.get()
                    img_data = img_data.astype(np.float32)
                    img_data = (img_data / 1e2) - 273
                    # todo: convert image from celcius to kelvin 100k
                    f.write(f"{ts} | {str(img_data.tolist())}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class FlirDevice(DeviceInterface):
    """
    Device implementation for FlirDevice
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

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
                                           self.logger)
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
                self.write_src = f"{self.run_config['experiment_dir']}/flir.csv"
                print(f"Write Source: {self.write_src}")
                self.writer = FlirWriterThread(self.sensor_queue,
                                               self.write_src,
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
