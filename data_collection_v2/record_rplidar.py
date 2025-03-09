"""
Main driver class for  rplidar for raspberry pi
Author: Prasoon Patidar
Created at: 5th Feb 2024
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
import numpy as np
import jstyleson as json
from datetime import datetime
import serial
from serial.tools import list_ports
from rplidar import RPLidar
from queue import Queue
from pathlib import Path
import signal
import pickle
import base64
import psutil
import subprocess

from utils import get_logger
from deviceInterface import DeviceInterface
from featurize.f1 import get_features
from database.DBO import DBO

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

# config parameters (not to be changed)
LIDAR2D_DEVICE_NAME = 'RPLidar'
LIDAR2D_PRODUCT_NAME = 'CP2102 USB to UART Bridge Controller'
RPLIDAR_GET_HEALTH_BYTE = b'\x52'
DESCRIPTOR_LEN = 7
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 20


class RPLidarReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr device
    """

    def __init__(self, out_queue, port_address, logger, feature_queue):
        threading.Thread.__init__(self)

        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.feature_queue = feature_queue
        self.port_address = port_address

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {LIDAR2D_DEVICE_NAME} sensor...")
        # start
        super(RPLidarReaderThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        lidar = RPLidar(self.port_address,logger=self.logger)
        while (True):
            try:
                lidar.start()
                print("Lidar initiation successful")
                # time.sleep(1)
                break
            except:
                print("Error in initiating lidar, trying again..")
                time.sleep(5)
            # lidar.stop()
            lidar.disconnect()
            lidar = RPLidar(self.port_address,logger=self.logger)
        lidar.disconnect()
        del lidar
        lidar = RPLidar(self.port_address, timeout=2,logger=self.logger)
        try:
            print('Recording measurements... Press Crl+C to stop.')
            for scan in lidar.iter_scans(scan_type='express', min_len=100, max_buf_meas=False):
                if not self.running:
                    break
                ts = time.time_ns()
                scan_x = []
                scan_y = []
                for obj in scan:
                    # print(f"Q: {obj[0]}, A: {obj[1]}, D: {obj[2]}")
                    scan_x.append(obj[2] * np.cos(np.radians(obj[1])))
                    scan_y.append(obj[2] * np.sin(np.radians(obj[1])))
                    # logging.info("(%.1f,%.1f)", scan_x[-1],scan_y[-1])
                self.out_queue.put((ts, (scan_x, scan_y)))
                if self.feature_queue is not None:
                    self.feature_queue.put((ts, (scan_x, scan_y)))
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)

class RPLidarWriterThread(threading.Thread):
    """
    Writer thread for doppler sensing from awr device
    """

    def __init__(self, in_queue, write_dir, start_hour, end_hour, logger, prefix='rplidar_2d', disk_limit=80):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_dir = write_dir
        self.running = False
        self.prefix = prefix
        self.out_file = None
        self.disk_limit=disk_limit
        self.start_hour = start_hour
        self.end_hour = end_hour        
        self.file_start_time = None
        self.checkpoint = time.time()
        self.ckpt_file = '/tmp/rplidar.ckpt'
        self.num_ckpt_frames = 0

    def start(self):
        # connect with device for reading data
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
        self.file_start_time = time.time()
        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {LIDAR2D_DEVICE_NAME} sensor...")
        # start
        super(RPLidarWriterThread, self).start()

    def renew_file(self):

        # release older b64
        self.out_file.close()

        # create new b64 based on timestamp of next frame and reset current frame number
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
        self.file_start_time = time.time()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        is_header_set = False
        with open(self.ckpt_file,'w') as ckpt_f:
            ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},0.0')
        try:
            # run till thread is running
            curr_hr = int(datetime.now().strftime("%H"))
            #check for disk fraction limit
            disk_cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
            disk_frac = subprocess.check_output(disk_cmd, shell=True).decode("utf-8")
            disk_frac = int(disk_frac.split("%")[0])    
            while self.running:
                # run till this video exhausts
                while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
                    if self.running:
                        ts, scan_data = self.in_queue.get()
                        if (curr_hr >= self.start_hour) and (curr_hr <= self.end_hour) and (disk_frac<=self.disk_limit):
                            encoded_data = base64.encodebytes(pickle.dumps(scan_data)).decode()
                            self.out_file.write(f"{ts} | {encoded_data} ||")
                            self.num_ckpt_frames+=1
                        else:
                            self.num_ckpt_frames=-CHECKPOINT_FREQ
                            if disk_frac>self.disk_limit:
                                self.num_ckpt_frames=-10*CHECKPOINT_FREQ
                        if time.time()-self.checkpoint>CHECKPOINT_FREQ:
                            with open(self.ckpt_file,'w') as ckpt_f:
                                ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},{round(self.num_ckpt_frames/CHECKPOINT_FREQ,2)}')
                            self.checkpoint = time.time()
                            self.num_ckpt_frames=0.
                    else:
                        self.out_file.close()
                        break
                if self.running:
                    curr_hr = int(datetime.now().strftime("%H"))
                    #check for disk fraction limit
                    disk_cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
                    disk_frac = subprocess.check_output(disk_cmd, shell=True).decode("utf-8")
                    disk_frac = int(disk_frac.split("%")[0])    
                    self.renew_file()
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class RPLidarDevice(DeviceInterface):
    """
    Device implementation for RPLidarDevice
    """

    def __init__(self, run_config, sensor_queue, logger, feature_queue):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, feature_queue)

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
            # self.reader = threading.Thread(target=readAndParseDataRPLidar,
            #                                args=(self.port_address, self.sensor_queue, self.feature_queue))
            self.reader = RPLidarReaderThread(self.sensor_queue,
                                              self.port_address,
                                              self.logger,
                                              self.feature_queue)
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
            self.writer = RPLidarWriterThread(self.sensor_queue,
                                              self.run_config['experiment_dir'],
                                              self.run_config['start_hour'],
                                              self.run_config['end_hour'],                                            
                                              self.logger,
                                              disk_limit=self.run_config['disk_limit'])
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
    logger = get_logger("rplidar",logdir = f'{Path(__file__).parent}/../../cache/logs',console_log=True)
    default_config_file = f'{Path(__file__).parent}/config.json'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")


    run_config = json.load(open(config_file, 'r'))
    
    # change config name to be rpi eth0 mac address
    # eth0_mac_cmd = "ifconfig eth0 | grep ether | awk 'END { print $2;}'"
    # mac_address = subprocess.check_output(eth0_mac_cmd,shell=True).decode('utf-8')
    # run_config['name']=f"rpi{mac_address.replace(':','')}".replace('\n','').replace('$','')

    # get experiment dir
    experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    run_config['experiment_dir'] = experiment_dir

    # initialize queues
    sensor_queue = Queue()
    feature_queue = Queue()
    
    #initialize DBHandler
    db_handler = DBO()
    if not db_handler.is_http_success(logger):
        logger.error("tsdb not working properly, exiting...")
        sys.exit(1)
    # initialize device
    lidar2dSensor = RPLidarDevice(run_config, sensor_queue, logger, feature_queue)

    # check if available
    if lidar2dSensor.is_available():
        logger.info(f"- Found Sensor {lidar2dSensor.name}-")
        lidar2dSensor.startReader()
        lidar2dSensor.startWriter()
    else:
        logger.error("Device not accessible, exiting...")
        sys.exit(1)

    # run for a given max duration
    try:
        num_frames = 0
        start_ft_time = time.time()
        frame_data = []
        window_name='lidar'
        curr_timestamp = start_ft_time
        while True:
            time.sleep(0.1) # DO NOT REMOVE THIS OR CODE WILL BREAK...
            if feature_queue.qsize() > 0:
                # logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                frame_time, scan = feature_queue.get()
                if frame_time // 10**9 > curr_timestamp:
                    curr_timestamp = frame_time // 10**9
                    if len(frame_data) > 0.:
                        try:
                            ts_values, features = get_features(frame_data,'rplidar')
                            logger.info(f"got features: {ts_values.shape}, {features.shape}")
                            db_handler.write_features(run_config['name'], 'rplidar', run_config["featurizer"],
                                                      ts_values, features)
                            # ts_values = ts_values*1000
                            # #logger.info(f"got features: {ts_values.shape}, {features.shape}")
                            # db_handler.write_features(run_config['name'], 'rplidar', run_config["featurizer"],
                            #                         ts_values.astype(np.int64), features)
                        except:
                            logger.warning("Error in writing features to TSDB")
                            logger.warning(traceback.format_exc())
                    #post_lidar_features(frame_data)
                    frame_data = []
                frame_data.append((frame_time, scan))
                num_frames += 1
            if time.time() > start_ft_time + 10.:
                logger.info(f"Num Lidar2D Frames in last 10 Secs: {num_frames}")
                num_frames = 0
                start_ft_time = time.time()
        
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        # cv2.destroyWindow(window_name)
        logger.info(f"Data Collection Complete {lidar2dSensor.name}")
    except KeyboardInterrupt:
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        logger.info(f"Stopped {lidar2dSensor.name}")
    except:
        logger.info(f"Exited featurization loop...")
        logger.info(traceback.format_exc())
    finally:
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        logger.info(f"Stopped {lidar2dSensor.name}")

