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
import json
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
from sensing.privacy_sensors.doppler.dca1000.serialConfig import configure_dca, configure_serial
from sensing.privacy_sensors.doppler.dca1000.serialConfig import stop_sensor, reset_dca
from sensing.privacy_sensors.doppler.dca1000.DCA1000 import DCA1000


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


# config variables
if True:
    # config parameters (not to be changed)
    AWR_DEVICE_NAME = 'AWR1642BOOST-ODS'
    BASE_LVDS_CONFIG = f'{Path(__file__).parent}/sensing/privacy_sensors/doppler/dca1000/lvds_config.json'
    BASE_LVDS_BINARY = f'{Path(__file__).parent}/DCA1000EVM_CLI_Control'
    # CLIPORT_ADDR = '/dev/ttyACM0'
    DEVICE_HWID = 'SER=R0061036'
    CLIPORT_HWID = '4.4.2:1.0'
    # 'USB VID:PID=0451:BEF3  LOCATION=1-3.4.4.4.2:1.0'
    CLIPORT_BAUDRATE = 115200
    # DATAPORT_ADDR = '/dev/ttyACM1'
    DATAPORT_HWID = '4.4.2:1.3'
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

    # Write COnstants
    MAX_DURATION_PER_FILE = 15 * 60
    CHECKPOINT_FREQ = 60

BASE_CONFIG = f'{Path(__file__).parent}/sensing/privacy_sensors/doppler/dca1000/f30_v3.cfg'

# f30_hot.cfg
# numChirpsPerFrame = 32
# numRxAntennas = 4
# numTxAntennas = 2
# numADCSamples = 272

# f30_v3.cfg
numChirpsPerFrame = 64
numRxAntennas = 4
numTxAntennas = 2
numADCSamples = 240

# f30_v4.cfg
# numChirpsPerFrame = 64
# numRxAntennas = 4
# numTxAntennas = 2
# numADCSamples = 240

# f20_v5.cfg
# numChirpsPerFrame = 32
# numRxAntennas = 4
# numTxAntennas = 2
# numADCSamples = 256


# f30_v6.cfg
# numChirpsPerFrame = 64
# numRxAntennas = 4
# numTxAntennas = 2
# numADCSamples = 128


if __name__ == '__main__':
    logger = get_logger("vax_doppler")
    default_config_file = 'config/dc_config.json'
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
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = open(f'{experiment_dir}/doppler_{time_str}.csv', 'a+')

    # get cliport address
    ports = serial.tools.list_ports.comports()
    cliport_address = None
    for port in ports:
        # print(port.hwid, port.device)
        if (CLIPORT_HWID in port.hwid) & (DEVICE_HWID in port.hwid):
            cliport_address = port.device
    if cliport_address is None:
        logger.error("CLI port not found for doppler, exiting")
        sys.exit(1)
    logger.info(f"Got CLI Port Address: {cliport_address}")
    # initialize device
    configure_serial(BASE_CONFIG, cliport_address)
    time.sleep(1)
    reset_dca(BASE_LVDS_BINARY, BASE_LVDS_CONFIG)
    time.sleep(1)
    configure_dca(BASE_LVDS_BINARY, BASE_LVDS_CONFIG)
    time.sleep(1)
    try:
        ADC_PARAMS = {'chirps': numChirpsPerFrame // numTxAntennas,
                      'rx': numRxAntennas,
                      'tx': numTxAntennas,
                      'samples': numADCSamples,
                      'IQ': 2,
                      'bytes': 2}
        dca = DCA1000(adc_params=ADC_PARAMS)
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, screen_width // 2, (2 * screen_height) // 5)
        num_frames = 0
        file_start_time = time.time()
        start_viz_time = time.time()
        checkpoint = time.time()
        ckpt_file = '/tmp/doppler.ckpt'
        while time.time() - start_time < max_duration:
            raw_adc_data = dca.read(timeout=.1)
            timestamp_ns = time.time_ns()
            radar_cube = dca.organize(raw_adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas,
                                      num_samples=numADCSamples)
            assert radar_cube.shape == (
                numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"
            fft1d_out = dsp.range_processing(radar_cube, window_type_1d=Window.HANNING)
            fft1d_out = dsp.compensation.clutter_removal(fft1d_out)
            det_matrix, aoa_input = dsp.doppler_processing(fft1d_out, num_tx_antennas=numTxAntennas,
                                                           clutter_removal_enabled=True,
                                                           window_type_2d=Window.HANNING, interleaved=False)
            out_data_dict = {'det_matrix': det_matrix, 'aoa_input': aoa_input}
            # det_matrix
            # b64_data = base64.encodebytes(pickle.loads(det_matrix)).decode('ascii')
            # encoded_matrix = base64.encodebytes(pickle.dumps(det_matrix)).decode()
            encoded_matrix = base64.encodebytes(pickle.dumps(out_data_dict)).decode()
            outfile.write(f"{timestamp_ns} | {encoded_matrix} ||")
            if visualize:
                det_matrix_vis = np.fft.fftshift(det_matrix[:,:32], axes=1)
                img = det_matrix_vis.T
                img = img.astype(np.float32)
                img = 255 * (img - img.min()) / (img.max() - img.min())

                img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                # cv2.imshow(window_name, img_col)
                img_col = cv2.fastNlMeansDenoisingColored(img_col, None, 10, 10, 7, 15)
                # resizing image for better visualization
                scale_percent = 700  # percent of original size
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
        stop_sensor(cliport_address)
        # time.sleep(0.5)
        reset_dca(BASE_LVDS_BINARY, BASE_LVDS_CONFIG)
    except:
        traceback.print_exc()
        outfile.close()
        stop_sensor(cliport_address)
        reset_dca(BASE_LVDS_BINARY, BASE_LVDS_CONFIG)
    finally:
        outfile.close()
        stop_sensor(cliport_address)
        reset_dca(BASE_LVDS_BINARY, BASE_LVDS_CONFIG)

