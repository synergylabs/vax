from run import readAndParseData1642Boost, stopDoppler, startDoppler
import traceback
import numpy as np
import cv2
import tempfile
import time
import serial
import sys
from pathlib import Path
from serial.tools import list_ports
awr6843_filename = tempfile.mktemp(prefix='delme_awr6843_unlimited_',
                                     suffix='.csv', dir='')
file_handler = open(awr6843_filename,'w')
AWR_DEVICE_NAME = 'AWR6843AOP'
# CLIPORT_ADDR = '/dev/ttyACM0'
DEVICE_HWID = 'SER=01083238'
CLIPORT_HWID = ':1.0'
# 'USB VID:PID=0451:BEF3  LOCATION=1-3.4.4.4.2:1.0'
CLIPORT_BAUDRATE = 115200
# DATAPORT_ADDR = '/dev/ttyACM1'
DATAPORT_HWID = ':1.1'
DATAPORT_BAUDRATE = 921600
BASE_CONFIG = f'{Path(__file__).parent}/RangeDopplerAOP.cfg'

# get cliport address
ports = serial.tools.list_ports.comports()
cliport_address = None
for port in ports:
    print(port.hwid, port.device)
    if (CLIPORT_HWID in port.hwid) & (DEVICE_HWID in port.hwid):
        cliport_address = port.device
if cliport_address is None:
    print("CLI port not found for doppler, exiting")
    sys.exit(1)
print(f"Got CLI Port Address: {cliport_address}")

dataport_address = None
for port in ports:
    print(port.hwid, port.device)
    if (DATAPORT_HWID in port.hwid) & (DEVICE_HWID in port.hwid):
        dataport_address = port.device
if cliport_address is None:
    print("Data port not found for doppler, exiting")
    sys.exit(1)
print(f"Got DATA Port Address: {dataport_address}")

cliport, dataport, configdict = startDoppler(cliport_address, dataport_address, BASE_CONFIG)
while True:
    try:
        dataOk, frameNumber, detObj = readAndParseData1642Boost(dataport, configdict)
        if dataOk:
            img =detObj['rangeDopplerMatrix']
            file_handler.write(f'{str(time.time_ns())} | {str(img)}')
            img = img.astype(np.float32)
            img = 255 * (img - img.min()) / (img.max() - img.min())
            img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

            # resizing image for better visualization
            scale_percent = 800  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Doppler', img_resized)
        if cv2.waitKey(1) == 27:
            print("Closing Doppler")
            stopDoppler(cliport, dataport)
            break  # esc to quit
    except:
        print("Closing Doppler")
        print(traceback.print_exc())
        stopDoppler(cliport,dataport)
