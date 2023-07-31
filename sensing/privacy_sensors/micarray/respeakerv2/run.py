"""
Main driver class for  micarray recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""
# basic libraries
from datetime import datetime
from queue import Queue
import json
import subprocess
import socket
import threading
import time
import traceback
import sys
import os
import pickle
import base64
import signal
import cv2
import numpy as np

from sensing.utils import get_logger, get_screen_size
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
RESPEAKER_DEVICE_NAME = 'micarrayv2'
ODAS_EXECUTABLE_PATH = '/home/vax/sensors/micarray/odas/build/bin/odaslive'
ODAS_CONFIG_PATH = '/home/vax/sensors/micarray/odas/build/bin/terminalv2.cfg'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 60

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


class Respeaker4MicReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, logger, viz_queue):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.running = False
        self.logger = logger
        self.viz_queue = viz_queue

    def start(self):

        # mark this is running
        self.running = True

        # start
        super(Respeaker4MicReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],stdout=subprocess.PIPE)
        try:
            store1, store2 = "", ""
            sst_ssl_ctr = 1
            num_frames_per_sec = 0
            curr_timestamp = time.time()
            while True:
                next_line = micarray_proc.stdout.readline().decode()[:-1]
                #print(next_line)
                if len(next_line) == 0:
                    break
                if sst_ssl_ctr == 1:
                    store1 += next_line
                    if next_line == '}':
                        sst_ssl_ctr = 2
                elif sst_ssl_ctr == 2:
                    store2 += next_line
                    if next_line == '}':
                        try:
                            store1_arr = json.loads(store1)['src']
                            store2_arr = json.loads(store2)['src']
                        except:
                            store1, store2, sst_ssl_ctr = "", "", 1
                            continue
                        num_frames_per_sec+=1
                        if ("E" in store1_arr[0].keys()) & ("id" in store2_arr[0].keys()):
                            detObj = {"SSL": store1_arr, "SST": store2_arr}
                        elif ("id" in store1_arr[0].keys()) & ("E" in store2_arr[0].keys()):
                            detObj = {"SSL": store2_arr, "SST": store1_arr}
                        else:
                            continue
                        time_val_ = time.time_ns()
                        self.out_queue.put((time_val_, detObj))
                        store1, store2, sst_ssl_ctr = "", "", 1
                        if time.time()-curr_timestamp > 1.:
                            self.viz_queue.put((num_frames_per_sec,time_val_,detObj))
                            curr_timestamp = time.time()
                            num_frames_per_sec = 0.
                else:
                    break
            self.logger.debug("Closing Process")
            micarray_proc.kill()
        except Exception as e:
            self.running = False
            micarray_proc.kill()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Respeaker4MicWriterThread(threading.Thread):
    """
    Writer thread for doppler sensing from awr device
    """

    def __init__(self, in_queue, write_dir, logger, prefix='micarrayv2'):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_dir = write_dir
        self.is_running = False
        self.prefix = prefix
        self.out_file = None
        self.file_start_time = None
        self.checkpoint = time.time()
        self.ckpt_file = '/tmp/micarray.ckpt'

    def start(self):
        # connect with device for reading data
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.csv', 'w')
        self.file_start_time = time.time()

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {RESPEAKER_DEVICE_NAME} sensor...")
        # start
        super(Respeaker4MicWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def renew_file(self):

        # release older csv
        self.out_file.close()

        # create new csv based on timestamp of next frame and reset current frame number
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.csv', 'w')
        # self.csv_out = csv.writer(self.out_file)
        self.file_start_time = time.time()

    def run(self):
        is_header_set = False
        try:
            # run till thread is running
            while self.running:
                # run till this video exhausts
                while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
                    if self.running:
                        time_val_, data_dict = self.in_queue.get()
                        encoded_data = base64.encodebytes(pickle.dumps(data_dict)).decode()
                        self.out_file.write(f"{time_val_} | {encoded_data} ||")
                        if time.time()-self.checkpoint>CHECKPOINT_FREQ:
                            with open(self.ckpt_file,'w') as ckpt_f:
                                ckpt_f.write(f'{datetime.now()}')
                            self.checkpoint = time.time()
                    else:
                        self.out_file.close()
                        break
                if self.running:
                    self.renew_file()

            with open(self.write_src, 'w') as f:
                while self.running:
                    time_val_, data_dict = self.in_queue.get()
                    # time_val_rpi = parser.parse(data_dict['Timestamp'])
                    # ts = int(float(time_val_rpi.strftime('%s.%f')) * 1e9)
                    encoded_data = base64.encodebytes(pickle.dumps(data_dict)).decode()
                    f.write(f"{time_val_} | {encoded_data} ||")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class Respeaker4MicDevice(DeviceInterface):
    """
    Device implementation for Respeaker4MicDevice
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

        # initialize Respeaker4MicDevice
        self.name = RESPEAKER_DEVICE_NAME

        self.rpi_address = ''
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],stdout=subprocess.PIPE)
        time.sleep(1)
        if micarray_proc.poll() is None:
            micarray_proc.kill()
            return True
        else:
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = Respeaker4MicReaderThread(self.sensor_queue,
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
                # time_str = datetime.now().strftime("%H%M%S")
                # self.write_src = f"{WRITE_PATH}/respeaker_{time_str}.csv"
                #print(f"Write Source: {self.write_src}")
                self.writer = Respeaker4MicWriterThread(self.sensor_queue,
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
    logger = get_logger("vax_micarrayv2")
    default_config_file = 'config/dc_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'Micarray'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins'] * 60
    t_data_collection_start = datetime.now()
    start_time = time.time()
    # get experiment dir
    experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}/{t_data_collection_start.strftime('%Y%m%d_%H')}"
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
    micarraySensor = Respeaker4MicDevice(run_config, sensor_queue, logger, viz_queue)

    # check if available
    if micarraySensor.is_available():
        logger.info(f"- Found Sensor {micarraySensor.name}-")
        micarraySensor.startReader()
        micarraySensor.startWriter()

    # run for a given max duration
    try:
        ID_IDX = "id"
        X_IDX, Y_IDX, Z_IDX = "x", "y", "z"
        ACT_IDX = "activity"
        colors = [(255, 0, 0), (0, 255, 0),
                  (0, 0, 255), (255, 255, 255)]
        micarray_frames = np.zeros((100, 2))
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, (3 * screen_width) // 4, (4 * screen_height) // 5)
        num_frames = 0
        start_viz_time = time.time()
        while (time.time() - start_time < max_duration):
            if visualize:
                if viz_queue.qsize() > 0:
                    num_frames_server, time_val_, detObj = viz_queue.get()
                    # print(num_frames_server, time_val_, detObj)
                    sst_arr = detObj['SST']
                    sst_dict = {val["id"]: val for val in sst_arr}
                    sources = []
                    for mic_src in sst_dict.keys():
                        if not (sst_dict[mic_src][ID_IDX] == 0):
                            sources.append(mic_src)

                    window_dimension = 400
                    multipler = window_dimension // 4
                    cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
                    cv2_img = cv2.line(cv2_img, (0, window_dimension // 2), (window_dimension, window_dimension // 2),
                                       (255, 255, 255), 4)
                    cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0), (window_dimension // 2, window_dimension),
                                       (255, 255, 255), 4)
                    for mic_src in sources:
                        idx, x, y, z = sst_dict[mic_src][ID_IDX], sst_dict[mic_src][X_IDX], sst_dict[mic_src][Y_IDX], \
                            sst_dict[mic_src][Z_IDX]
                        px = min((window_dimension // 2) - int(x * multipler), window_dimension)
                        py = min((window_dimension // 2) + int(y * multipler), window_dimension)
                        z_size = int(np.abs(z) * 10)
                        z_sign = np.sign(z)
                        if z_sign < 0:
                            cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                        else:
                            cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                        # print(px,py,z_size)

                    img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_OCEAN)
                    # print(img_col.shape)
                    scale_percent = 100  # percent of original size
                    width = int(cv2_img.shape[1] * scale_percent / 100)
                    height = int(cv2_img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    # print(img_resized.shape)
                    cv2.imshow('Micarray', img_resized)
                    if cv2.waitKey(1) == 27:
                        print("Closing Micarray")
                        # stop2DLidar(serialObj)
                        break  # esc to quit
                    num_frames += num_frames_server
                if time.time() > start_viz_time + 10.:
                    logger.info(f"Num Micarray Frames in last 10 Secs: {num_frames}")
                    num_frames = 0
                    start_viz_time = time.time()
        micarraySensor.stopWriter()
        micarraySensor.stopReader()
        # cv2.destroyWindow(window_name)
        logger.info(f"Stopped {micarraySensor.name}")
    except KeyboardInterrupt:
        micarraySensor.stopWriter()
        micarraySensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {micarraySensor.name}")
    finally:
        micarraySensor.stopWriter()
        micarraySensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Data Collection Complete {micarraySensor.name}")

