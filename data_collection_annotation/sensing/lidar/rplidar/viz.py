#!/usr/bin/env python3
'''Records scans to a given file in the form of numpy array.
Usage example:

$ ./record_scans.py out.npy'''
import sys
from rplidar import RPLidar
import time
import numpy as np
import serial
from serial.tools import list_ports
import queue
import threading
import logging
import cv2

scan_queue = queue.Queue()


def readAndParseDataRPLidar(port_address, squeue):
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
    lidar = RPLidar(port_address)
    try:

        print('Recording measurements... Press Crl+C to stop.')
        for scan in lidar.iter_scans():
            scan_x = []
            scan_y = []
            for obj in scan:
                # print(f"Q: {obj[0]}, A: {obj[1]}, D: {obj[2]}")
                scan_x.append(obj[2] * np.cos(np.radians(obj[1])))
                scan_y.append(obj[2] * np.sin(np.radians(obj[1])))
                # logging.info("(%.1f,%.1f)", scan_x[-1],scan_y[-1])
            squeue.put((scan_x, scan_y))
    except KeyboardInterrupt:
        print('Stopping.')

    lidar.stop()
    time.sleep(1)
    lidar.disconnect()

import tempfile
import time
rplidar_filename = tempfile.mktemp(prefix='delme_rplidar_unlimited_',
                                     suffix='.csv', dir='')
file_handler = open(rplidar_filename,'w')
# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__ == '__main__':
    product_name = 'CP2102 USB to UART Bridge Controller'
    ports = serial.tools.list_ports.comports()
    port_address = ''
    for port in ports:
        if port.product == product_name:
            port_address = port.device
    if port_address == '':
        print("Device not found. Exiting.")
        sys.exit(1)
    print(f"Lidar found at port: {port_address}")
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=readAndParseDataRPLidar, args=(port_address, scan_queue))
    logging.info("Main    : before running thread")
    x.start()
    # fig, axs = plt.subplots(1,1)
    #
    # plt.show()
    #

    while True:
        scan_x, scan_y = scan_queue.get()
        file_handler.write(f"{str(time.time_ns())}, {str(scan_x)}, {str(scan_y)}")
        # print(scan_x[:10], scan_y[:10])
        window_dimension = 400
        divider = 8000 // window_dimension
        cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
        cv2_img = cv2.line(cv2_img, (0, window_dimension // 2), (window_dimension, window_dimension // 2),
                           (255, 255, 255), 4)
        cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0), (window_dimension // 2, window_dimension),
                           (255, 255, 255), 4)
        for x, y in zip(scan_x, scan_y):
            if (np.abs(x) // divider < window_dimension) & (np.abs(y) // divider < window_dimension):
                px = min((window_dimension // 2) + int(x // divider), window_dimension)
                py = min((window_dimension // 2) + int(y // divider), window_dimension)
                cv2_img = cv2.circle(cv2_img, (px, py), divider // 8, (255, 255, 255), -1)
        img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_BONE)
        # print(img_col.shape)
        scale_percent = 100  # percent of original size
        width = int(cv2_img.shape[1] * scale_percent / 100)
        height = int(cv2_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
        # print(img_resized.shape)
        cv2.imshow('2D Lidar', img_resized)
        if cv2.waitKey(1) == 27:
            print("Closing 2D Lidar")
            # stop2DLidar(serialObj)
            break  # esc to quit

    #
    #     axs.clear()
    #     axs.scatter(scan_x,scan_y)
    #     plt.show()
    #     if cv2.waitKey(1) == 27:
    #         print("Closing 2D Lidar")
    #         break
