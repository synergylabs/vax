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

from utils import get_logger
from deviceInterface import DeviceInterface

# config parameters (not to be changed)
RESPEAKER_DEVICE_NAME = 'micarray'
WRITE_PATH = '/home/pi/micarray_data'
ODAS_EXECUTABLE_PATH = '/home/pi/odas/build/bin/odaslive'
ODAS_CONFIG_PATH = '/home/pi/odas/build/bin/terminal.cfg'


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
            self.logger.debug("Closing SSH")
            micarray_proc.kill()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Respeaker4MicWriterThread(threading.Thread):
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
        self.logger.info(f"Starting writing data from {RESPEAKER_DEVICE_NAME} sensor...")
        # start
        super(Respeaker4MicWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        is_header_set = False
        try:
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
                time_str = datetime.now().strftime("%H%M%S")
                self.write_src = f"{WRITE_PATH}/respeaker_{time_str}.csv"
                #print(f"Write Source: {self.write_src}")
                self.writer = Respeaker4MicWriterThread(self.sensor_queue,
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


if __name__ == '__main__':
    logger = get_logger("vax_micarray_serverside")
    visualize = True
    window_name = 'Micarray'
    t_data_collection_start = datetime.now()
    start_time = time.time()

    # initialize queues
    sensor_queue = Queue()
    if visualize:
        viz_queue = Queue()
    else:
        viz_queue = None
    # initialize device
    micarraySensor = Respeaker4MicDevice({}, sensor_queue, logger, viz_queue)

    # check if available
    if micarraySensor.is_available():
        logger.info(f"- Found Sensor {micarraySensor.name}-")
        micarraySensor.startReader()
        micarraySensor.startWriter()

    # run for a given max duration
    try:
        while True:
            if visualize:                
                num_frames, time_val_, detObj = viz_queue.get()
                print(json.dumps([num_frames,time_val_,detObj]))
        micarraySensor.stopWriter()
        micarraySensor.stopReader()
        logger.info(f"Data Collection Complete {micarraySensor.name}")
    except KeyboardInterrupt:
        micarraySensor.stopWriter()
        micarraySensor.stopReader()
        logger.info(f"Stopped {micarraySensor.name}")
