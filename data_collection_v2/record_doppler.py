"""
Main driver class for  doppler recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

import cv2

# basic libraries
import threading
import matplotlib
matplotlib.use('agg')
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
import matplotlib.pyplot as plt
import psutil

# custom libraries
from deviceInterface import DeviceInterface
from utils import get_logger, get_screen_size
from doppler.run import readAndParseData1642Boost, stopDoppler, startDoppler
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
CHECKPOINT_FREQ = 20
#checkpoint management
checkpoint = time.time()
ckpt_file = '/tmp/doppler.ckpt'
num_ckpt_frames = 0

BASE_CONFIG = f'{Path(__file__).parent}/doppler/RangeDopplerHeatmapV2.cfg'
# BASE_CONFIG =  f'{Path(__file__).parent}/doppler/configs/test.cfg'

if __name__ == '__main__':
    logger = get_logger("doppler", logdir=f'{Path(__file__).parent}/../../cache/logs', console_log=True)
    default_config_file = f'{Path(__file__).parent}/config.json'
    visualize = False
    if visualize:
        screen_width, screen_height = get_screen_size()
        window_name = 'Range-Doppler Heatmap'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
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
    with open(ckpt_file, 'w') as ckpt_f:
        ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},0.0')

    cliport, dataport_doppler, configdict = startDoppler(cliport_address, dataport_address, BASE_CONFIG)
    time.sleep(1)
    try:
        if visualize:
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, screen_width // 2, (2 * screen_height) // 5)
        num_frames = 0
        file_start_time = time.time()
        start_viz_time = time.time()
        prev_det_matrix = None
        while True:
            dataOk, frameNumber, detObj = readAndParseData1642Boost(dataport_doppler, configdict)
            if dataOk:
                timestamp_ns = time.time_ns()
                if 'rangeDopplerMatrix' not in detObj:
                    if prev_det_matrix is not None:
                        detObj['det_matrix'] = prev_det_matrix
                    else:
                        continue
                else:
                    prev_det_matrix = detObj['rangeDopplerMatrix']
                    detObj['det_matrix'] = detObj['rangeDopplerMatrix']
                    del detObj['rangeDopplerMatrix']
                encoded_matrix = base64.encodebytes(pickle.dumps(detObj)).decode()
                outfile.write(f"{timestamp_ns} | {encoded_matrix} ||")
                num_frames += 1
                num_ckpt_frames += 1
                # renew file every max frames
                if time.time() - file_start_time > MAX_DURATION_PER_FILE:
                    outfile.close()
                    file_start_time = time.time()
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    outfile = open(f'{experiment_dir}/doppler_{time_str}.csv', 'a+')

                if time.time() > start_viz_time + 10.:
                    logger.info(f"Num Frames in last 10 Secs: {num_frames}")
                    num_frames = 0
                    start_viz_time = time.time()

                if time.time() - checkpoint > CHECKPOINT_FREQ:
                    with open(ckpt_file, 'w') as ckpt_f:
                        ckpt_f.write(
                            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},{round(num_ckpt_frames / CHECKPOINT_FREQ, 2)}')
                    checkpoint = time.time()
                    num_ckpt_frames = 0.

                if visualize:
                    img = detObj['det_matrix']
                    img = img.astype(np.float32)
                    noise_red_data = dsp.compensation.clutter_removal(img).T
                    noise_red_data[noise_red_data < 0] = 0.
                    img = noise_red_data.T

                    img = 255 * (img - img.min()) / (img.max() - img.min())
                    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

                    # resizing image for better visualization
                    scale_percent = 800  # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_name, img_resized)
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
