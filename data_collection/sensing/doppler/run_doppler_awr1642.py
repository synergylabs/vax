"""
Main driver class for  doppler recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

import cv2

# basic libraries
import threading
import time
import logging
import pickle
import base64
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
import time
from queue import Queue
from datetime import datetime
import jstyleson as json
import pickle
import base64
# import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
import mmwave.dsp as dsp
import serial
from serial.tools import list_ports
import psutil

# custom libraries
from sensing.deviceInterface import DeviceInterface
from sensing.utils import get_logger, get_screen_size
from awr1642boost.run import readAndParseData1642Boost, stopDoppler, startDoppler
import traceback
import numpy as np
import cv2
import tempfile
import time


# from dca1000.serialConfig import configure_dca, configure_serial
# from dca1000.serialConfig import stop_sensor, reset_dca
# from dca1000.DCA1000 import DCA1000


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)

# config parameters (not to be changed)
AWR_DEVICE_NAME = 'AWR1642BOOST-ODS'
# CLIPORT_ADDR = '/dev/ttyACM0'
DEVICE_HWID = 'SER=R0061036'
CLIPORT_HWID = ':1.0'
# 'USB VID:PID=0451:BEF3  LOCATION=1-3.4.4.4.2:1.0'
CLIPORT_BAUDRATE = 115200
# DATAPORT_ADDR = '/dev/ttyACM1'
DATAPORT_HWID = ':1.3'
DATAPORT_BAUDRATE = 921600
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 60

BASE_CONFIG = f'{Path(__file__).parent}/awr1642boost/RangeDopplerHeatmap.cfg'

if __name__ == '__main__':
    logger = get_logger("vax_doppler")
    default_config_file = 'config/data_collection_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'Range-Doppler Heatmap'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins'] * 60
    t_data_collection_start = datetime.now()
    start_time = time.time()

    experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    run_config['experiment_dir'] = experiment_dir
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = open(f'{experiment_dir}/doppler_{time_str}.csv', 'a+')

    # get cliport address
    ports = serial.tools.list_ports.comports()
    cliport_address = None
    for port in ports:
        print(port.hwid, port.device)
        if (CLIPORT_HWID in port.hwid) & (DEVICE_HWID in port.hwid):
            cliport_address = port.device
    if cliport_address is None:
        logger.error("CLI port not found for doppler, exiting")
        sys.exit(1)
    logger.info(f"Got CLI Port Address: {cliport_address}")

    dataport_address = None
    for port in ports:
        print(port.hwid, port.device)
        if (DATAPORT_HWID in port.hwid) & (DEVICE_HWID in port.hwid):
            dataport_address = port.device
    if cliport_address is None:
        logger.error("Data port not found for doppler, exiting")
        sys.exit(1)
    logger.info(f"Got DATA Port Address: {dataport_address}")

    # initialize device

    cliport, dataport_doppler, configdict = startDoppler(cliport_address, dataport_address, BASE_CONFIG)
    time.sleep(1)
    try:
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, screen_width // 2, (2 * screen_height) // 5)
        num_frames = 0
        file_start_time = time.time()
        start_viz_time = time.time()
        checkpoint = time.time()
        ckpt_file = '/tmp/doppler.ckpt'
        while time.time() - start_time < max_duration:
            dataOk, frameNumber, detObj = readAndParseData1642Boost(dataport_doppler, configdict)
            if dataOk:
                timestamp_ns = time.time_ns()
                det_matrix, aoa_input = detObj['rangeDopplerMatrix'], None
                out_data_dict = {'det_matrix': det_matrix, 'aoa_input': aoa_input}
                encoded_matrix = base64.encodebytes(pickle.dumps(out_data_dict)).decode()
                outfile.write(f"{timestamp_ns} | {encoded_matrix} ||")
                if visualize:
                    img = det_matrix
                    img = img.astype(np.float32)
                    img = 255 * (img - img.min()) / (img.max() - img.min())
                    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

                    # resizing image for better visualization
                    scale_percent = 800  # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_name, img_resized)
                num_frames += 1
                if time.time() > start_viz_time + 10.:
                    logger.info(f"Num Frames in last 10 Secs: {num_frames}")
                    num_frames = 0
                    start_viz_time = time.time()
                if time.time() - checkpoint > CHECKPOINT_FREQ:
                    with open(ckpt_file, 'w') as ckpt_f:
                        ckpt_f.write(f'{datetime.now()}')
                    checkpoint = time.time()
                # renew file every max frames
                if time.time() - file_start_time > MAX_DURATION_PER_FILE:
                    outfile.close()
                    file_start_time = time.time()
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    outfile = open(f'{experiment_dir}/doppler_{time_str}.csv', 'a+')

                # logger.info(f"Frame Index: {frame_idx}")
                # time.sleep(0.5)
                if cv2.waitKey(1) == 27:
                    logger.info("Closing Doppler")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        outfile.close()
        stopDoppler(cliport, dataport_doppler)
    except:
        traceback.print_exc()
        outfile.close()
        stopDoppler(cliport, dataport_doppler)
    # finally:
    #     outfile.close()
    #     stopDoppler(cliport, dataport)
