"""
Main driver class for  lidar2d recording
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
import numpy as np
import json
from datetime import datetime
import serial
from serial.tools import list_ports
from rplidar import RPLidar
from queue import Queue
import signal
import pickle
import base64
import psutil

# custom libraries
from sensing.deviceInterface import DeviceInterface
from sensing.utils import get_logger, get_screen_size

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
CHECKPOINT_FREQ = 60

def readAndParseDataRPLidar(port_address, squeue, vizqueue):
    lidar = RPLidar(port_address)
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
        lidar = RPLidar(port_address)
    lidar.disconnect()
    del lidar
    lidar = RPLidar(port_address, timeout=2)
    try:

        print('Recording measurements... Press Crl+C to stop.')
        for scan in lidar.iter_scans(scan_type='express', min_len=100, max_buf_meas=False):
            ts = time.time_ns()
            scan_x = []
            scan_y = []
            for obj in scan:
                # print(f"Q: {obj[0]}, A: {obj[1]}, D: {obj[2]}")
                scan_x.append(obj[2] * np.cos(np.radians(obj[1])))
                scan_y.append(obj[2] * np.sin(np.radians(obj[1])))
                # logging.info("(%.1f,%.1f)", scan_x[-1],scan_y[-1])
            squeue.put((ts, (scan_x, scan_y)))
            vizqueue.put((ts, (scan_x, scan_y)))
    except KeyboardInterrupt:
        print('Stopping.')

    lidar.stop()
    time.sleep(1)
    lidar.disconnect()


class RPLidarReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr device
    """

    def __init__(self, out_queue, port_address, logger, viz_queue):
        threading.Thread.__init__(self)

        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.viz_queue = viz_queue
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
        lidar = RPLidar(self.port_address)
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
            lidar = RPLidar(self.port_address)
        lidar.disconnect()
        del lidar
        lidar = RPLidar(self.port_address, timeout=2)
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
                if self.viz_queue is not None:
                    self.viz_queue.put((ts, (scan_x, scan_y)))
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class RPLidarWriterThread(threading.Thread):
    """
    Writer thread for doppler sensing from awr device
    """

    def __init__(self, in_queue, write_dir, logger, prefix='rplidar'):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_dir = write_dir
        self.running = False
        self.prefix = prefix
        self.out_file = None
        self.file_start_time = None
        self.checkpoint = time.time()
        self.ckpt_file = '/tmp/lidar2d.ckpt'

    def start(self):
        # connect with device for reading data
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.csv', 'w')
        self.file_start_time = time.time()
        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {LIDAR2D_DEVICE_NAME} sensor...")
        # start
        super(RPLidarWriterThread, self).start()

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
        is_header_set = False
        try:
            # run till thread is running
            while self.running:
                # run till this video exhausts
                while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
                    if self.running:
                        ts, scan_data = self.in_queue.get()
                        encoded_data = base64.encodebytes(pickle.dumps(scan_data)).decode()
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
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class RPLidarDevice(DeviceInterface):
    """
    Device implementation for RPLidarDevice
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

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
            #                                args=(self.port_address, self.sensor_queue, self.viz_queue))
            self.reader = RPLidarReaderThread(self.sensor_queue,
                                              self.port_address,
                                              self.logger,
                                              self.viz_queue)
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
                # time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # self.write_prefix = f"{self.run_config['experiment_dir']}/rplidar_{time_str}.csv"
                # print(f"Write Source: {self.write_src}")
                self.writer = RPLidarWriterThread(self.sensor_queue,
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
    logger = get_logger("vax_lidar2d")
    default_config_file = 'config/dc_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'RPLidar'
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
    lidar2dSensor = RPLidarDevice(run_config, sensor_queue, logger, viz_queue)

    # check if available
    if lidar2dSensor.is_available():
        logger.info(f"- Found Sensor {lidar2dSensor.name}-")
        lidar2dSensor.startReader()
        lidar2dSensor.startWriter()

    # run for a given max duration
    try:
        lidar2d_frames = np.zeros((100, 2))
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, screen_width // 2, (2 * screen_height) // 5)
        num_frames = 0
        start_viz_time = time.time()
        while (time.time() - start_time < max_duration):
            if visualize:
                if viz_queue.qsize() > 0:
                    # logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                    frame_time, scan = viz_queue.get()
                    scan_x, scan_y = scan
                    # print(scan_x[:10], scan_y[:10])
                    window_dimension = 800
                    divider = 8000 // window_dimension
                    cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
                    # cv2_img = cv2.line(cv2_img, (0, window_dimension // 2), (window_dimension, window_dimension // 2),
                    #                    (255, 255, 255), 4)
                    # cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0), (window_dimension // 2, window_dimension),
                    #                    (255, 255, 255), 4)
                    for x, y in zip(scan_x, scan_y):
                        if (np.abs(x) // divider < window_dimension) & (np.abs(y) // divider < window_dimension):
                            px = min((window_dimension // 2) + int(x // divider), window_dimension)
                            py = min((window_dimension // 2) + int(y // divider), window_dimension)
                            cv2_img = cv2.circle(cv2_img, (px, py), divider // 8, (255, 255, 255), -1)
                    img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_BONE)
                    # print(img_col.shape)
                    scale_percent = 150  # percent of original size
                    width = int(cv2_img.shape[1] * scale_percent / 100)
                    height = int(cv2_img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    # print(img_resized.shape)
                    cv2.imshow(window_name, img_resized)
                    if cv2.waitKey(1) == 27:
                        print("Closing 2D Lidar")
                        break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    num_frames += 1
                if time.time() > start_viz_time + 10.:
                    logger.info(f"Num Lidar2D Frames in last 10 Secs: {num_frames}")
                    num_frames = 0
                    start_viz_time = time.time()
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        # cv2.destroyWindow(window_name)
        logger.info(f"Data Collection Complete {lidar2dSensor.name}")
    except KeyboardInterrupt:
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {lidar2dSensor.name}")
    finally:
        lidar2dSensor.stopWriter()
        lidar2dSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {lidar2dSensor.name}")

