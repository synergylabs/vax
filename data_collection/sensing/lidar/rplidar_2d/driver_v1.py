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
from serial.tools import list_ports
from rplidar import RPLidar
import os
import socket
import pickle
from pathlib import Path
import tempfile
import nmap
from dateutil import parser

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
LIDAR2D_DEVICE_NAME = 'RPLidar'
LIDAR2D_PRODUCT_NAME = 'CP2102 USB to UART Bridge Controller'
RPLIDAR_GET_HEALTH_BYTE = b'\x52'
DESCRIPTOR_LEN = 7

class RPLidarReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, port_address, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.port_address = port_address
        self.logger = logger
        self.running = False
        self.lidar = None

    def start(self):
        # mark this is running
        self.running = True

        # start
        super(RPLidarReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        while True:
            self.lidar = RPLidar(self.port_address)
            self.lidar._send_cmd(RPLIDAR_GET_HEALTH_BYTE)
            descriptor = self.lidar._serial.read(DESCRIPTOR_LEN)
            if len(descriptor) != DESCRIPTOR_LEN:
                self.logger.info(f"Descriptor = {str(descriptor)}")
                self.lidar.stop()
                self.lidar.disconnect()
                time.sleep(2)
            else:
                self.logger.info("RPLidar activation successful..")
                self.lidar.stop()
                self.lidar.disconnect()
                # self.lidar.clean_input()
                # self.lidar.start()
                break
        self.lidar = RPLidar(self.port_address)
        try:
            for scanframe in self.lidar.iter_scans():
                # self.logger.info(scanframe)
                scan_x = []
                scan_y = []
                for obj in scanframe:
                    # print(f"Q: {obj[0]}, A: {obj[1]}, D: {obj[2]}")
                    scan_x.append(obj[2] * np.cos(np.radians(obj[1])))
                    scan_y.append(obj[2] * np.sin(np.radians(obj[1])))
                    # logging.info("(%.1f,%.1f)", scan_x[-1],scan_y[-1])
                if self.running:
                    # scanframe = next(self.scan_iterator)
                    ts = time.time_ns()
                    self.out_queue.put((ts, (scan_x,scan_y)))
                else:
                    break
            self.lidar.disconnect()
        except Exception as e:
            self.running = False
            self.lidar.disconnect()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class RPLidarWriterThread(threading.Thread):
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
        self.logger.info(f"Starting writing data from {LIDAR2D_DEVICE_NAME} sensor...")
        # start
        super(RPLidarWriterThread, self).start()

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
                    ts, scan_data = self.in_queue.get()
                    f.write(f"{ts} | {str(scan_data)}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class RPLidarDevice(DeviceInterface):
    """
    Device implementation for RPLidarDevice
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

        # initialize RPLidarDevice
        self.name = LIDAR2D_DEVICE_NAME
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        ports = serial.tools.list_ports.comports()
        port_address = ''
        for port in ports:
            if port.product == LIDAR2D_PRODUCT_NAME:
                port_address = port.device
        if port_address == '':
            self.logger.error(f"Device {LIDAR2D_DEVICE_NAME} not found..")
            return False
        self.logger.info(f"Lidar found at port: {port_address}")
        self.port_address = port_address
        return True

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = RPLidarReaderThread(self.sensor_queue,
                                              self.port_address,
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
                self.write_src = tempfile.mktemp(prefix='delme_rplidar_rec_unlimited_',
                                                 suffix='.csv', dir='')
                print(f"Write Source: {self.write_src}")
                self.writer = RPLidarWriterThread(self.sensor_queue,
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
