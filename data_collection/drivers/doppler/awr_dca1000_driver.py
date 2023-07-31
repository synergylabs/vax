"""
Main driver class for doppler data collection from awr1642boost-ods using DCA Board
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
import serial
import os
from pathlib import Path
import tempfile
import subprocess
import signal
import mmwave as mm
from mmwave.dataloader import DCA1000, parse_raw_adc
import time
import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
import mmwave.dsp as dsp

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
AWR_DEVICE_NAME = 'AWR1642BOOST-ODS'
BASE_CONFIG = f'{Path(__file__).parent}/f30_hot.cfg'
BASE_LVDS_CONFIG = f'{Path(__file__).parent}/lvds_config.json'
BASE_LVDS_BINARY = f'{Path(__file__).parent}/DCA1000EVM_CLI_Control'
CLIPORT_ADDR = '/dev/ttyACM0'
CLIPORT_BAUDRATE = 115200
DATAPORT_ADDR = '/dev/ttyACM1'
DATAPORT_BAUDRATE = 921600

# Constants
OBJ_STRUCT_SIZE_BYTES = 12
BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
MMW_OUTPUT_MSG_DETECTED_POINTS = 1
MMW_OUTPUT_MSG_RANGE_PROFILE = 2
MMW_OUTPUT_MSG_NOISE_PROFILE = 3
MMW_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
MMW_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
MMW_OUTPUT_MSG_STATS = 6
maxBufferSize = 2 ** 15
tlvHeaderLengthInBytes = 8
pointLengthInBytes = 16
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]


class DCA1000ReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False

        #
        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.config_parameters = {}

        self.device_config = BASE_CONFIG
        self.dca_config = BASE_LVDS_CONFIG
        self.dca_binary = BASE_LVDS_BINARY
        self.dca = None  # object to fetch data from socket

    def configure_serial(self, configFileName):
        cliport = serial.Serial(CLIPORT_ADDR, CLIPORT_BAUDRATE)

        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            cliport.write((i + '\n').encode())
            self.logger.info(i)
            time.sleep(0.01)
        return cliport

    def configure_dca(self, dcaBinaryPath, ldvsConfigFilePath):
        # Check status of board

        response = subprocess.run([dcaBinaryPath, 'query_sys_status', ldvsConfigFilePath], capture_output=True)
        success_response = "System is connected."
        if success_response in response.stdout.decode():
            self.logger.info("DCA Board connected...")
        else:
            self.logger.info("DCA Board disconnected. Exiting.")
            sys.exit(1)

        # Setup fpga and record setting with config
        response = subprocess.run([dcaBinaryPath, 'fpga', ldvsConfigFilePath], capture_output=True)
        success_response = 'FPGA Configuration command : Success'
        if success_response in response.stdout.decode():
            self.logger.info("DCA FPGA Setup complete...")
        else:
            self.logger.info("DCA FPGA setup error. Exiting.", response.stderr.decode())
            sys.exit(1)

        response = subprocess.run([dcaBinaryPath, 'record', ldvsConfigFilePath], capture_output=True)
        success_response = 'Configure Record command : Success'
        if success_response in response.stdout.decode():
            self.logger.info("DCA LVDS Recording Setup complete...")
        else:
            self.logger.info("DCA LVDS Recording setup error. Exiting.", response.stderr.decode())
            sys.exit(1)

        response = subprocess.run([dcaBinaryPath, 'record', ldvsConfigFilePath], capture_output=True)
        success_response = 'Configure Record command : Success'
        if success_response in response.stdout.decode():
            self.logger.info("DCA LVDS Recording Setup complete...")
        else:
            self.logger.info("DCA LVDS Recording setup error. Exiting.", response.stderr.decode())
            sys.exit(1)

        recording_process = subprocess.Popen([dcaBinaryPath, 'start_record', ldvsConfigFilePath],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
        stdout, stderr = recording_process.communicate()
        success_response = 'Start Record command : Success'
        if success_response in stdout.decode():
            self.logger.info("DCA Recording Started Successfully...")
        else:
            self.logger.info("DCA Recording start error. Exiting.", stderr.decode())
            sys.exit(1)
        time.sleep(2)

        # set stop_record status to allow fpga reset in future
        response = subprocess.run([dcaBinaryPath, 'stop_record', ldvsConfigFilePath], capture_output=True)

        # get pid for spawned child process
        running_processes = subprocess.check_output(['ps', 'aux']).decode().split("\n")
        dca_processes = [xr for xr in running_processes if (f"start_record {ldvsConfigFilePath}" in xr)]
        for dca_process in dca_processes:
            dca_process_info = dca_process.split(" ")[1:]
            dca_process_pid = None
            for process_info_str in dca_process_info:
                if not process_info_str == '':
                    dca_process_pid = int(process_info_str)
                    break
            self.logger.info(f"DCA Recording Subprocess Spawned, PID: {dca_process_pid}")
            os.kill(dca_process_pid, signal.SIGKILL)
            self.logger.info(f"DCA Recording Subprocess killed: {dca_process_pid}")
        return None

    def stop_sensor(self):
        CLIport = serial.Serial(CLIPORT_ADDR, CLIPORT_BAUDRATE)
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        logger.info("Sensor Stopped...")
        return None

    def reset_dca(self, dcaBinaryPath, ldvsConfigFilePath):
        response = subprocess.run([dcaBinaryPath, 'reset_fpga', ldvsConfigFilePath], capture_output=True)
        success_response = 'Reset FPGA command : Success'
        if success_response in response.stdout.decode():
            self.logger.info("DCA Board Reset complete...")
        else:
            self.logger.info("DCA Board Reset error. Exiting.", response.stderr.decode())
            sys.exit(1)
        return

    def start(self):

        # parse config parameters
        self.configure_serial(self.device_config)
        time.sleep(2)
        self.configure_dca(self.dca_binary, self.dca_config)
        time.sleep(1)
        self.dca = DCA1000()
        # mark this is running
        self.running = True

        # start
        super(DCA1000ReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        try:
            while self.running:
                raw_adc_data = self.dca.read()
                timestamp_ns = time.time_ns()
                self.out_queue.put((timestamp_ns, raw_adc_data))
            self.stop_sensor()
            self.reset_dca(self.dca_binary, self.dca_config)
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            self.stop_sensor()
            self.reset_dca(self.dca_binary, self.dca_config)
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class DCA1000WriterThread(threading.Thread):
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
        self.logger.info(f"Starting writing data from {AWR_DEVICE_NAME} sensor...")
        # start
        super(DCA1000WriterThread, self).start()

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
                    ts, data_dict = self.in_queue.get()
                    f.write(f"{ts} | {data_dict}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class DCA1000Device(DeviceInterface):
    """
    Device implementation for DCA1000Device
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

        # initialize DCA1000Device
        self.name = AWR_DEVICE_NAME
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        try:
            test_cliport = serial.Serial(CLIPORT_ADDR, CLIPORT_BAUDRATE)
            test_dataport = serial.Serial(DATAPORT_ADDR, DATAPORT_BAUDRATE)
            test_cliport.close()
            test_dataport.close()
            return True
        except:
            self.logger.error(f"Unable to reach {AWR_DEVICE_NAME}: {traceback.format_exc()}")
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = DCA1000ReaderThread(self.sensor_queue,
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
                self.write_src = f"{self.run_config['experiment_dir']}/doppler.csv"
                self.logger.info(f"Write Source: {self.write_src}")
                self.writer = DCA1000WriterThread(self.sensor_queue,
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
