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
import matplotlib.pyplot as plt

def readAndParseDataRPLidar(port_address):
    lidar = RPLidar(port_address)
    frameData = []
    numFrames = 0
    st_time = time.time()
    while(True):
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
            frameData.append(np.array(scan))
            scan_x = []
            scan_y = []
            for obj in scan:
                # print(f"Q: {obj[0]}, A: {obj[1]}, D: {obj[2]}")
                scan_x.append(obj[2]*np.cos(np.radians(obj[1])))
                scan_y.append(obj[2]*np.sin(np.radians(obj[1])))
                # print(scan_x[-1],scan_y[-1])
            numFrames+=1
            if time.time() - st_time > 10.0:
                print(numFrames)
                numFrames = 0
                frameData = []
                st_time = time.time()
    except KeyboardInterrupt:
        print('Stopping.')

    lidar.stop()
    time.sleep(1)
    lidar.disconnect()

def getLidarPort():
    product_name = 'CP2102 USB to UART Bridge Controller'
    ports = serial.tools.list_ports.comports()
    port_address = ''
    for port in ports:
        if port.product==product_name:
            port_address = port.device
    if port_address=='':
        print("Device not found. Exiting.")
        sys.exit(1)
    print(f"Lidar found at port: {port_address}")
    return port_address

def start2DLidar():
    port_address = getLidarPort()
    lidar = RPLidar(port_address)
    while(True):
        try:
            lidar.start()
            print("Lidar initiation successful")
            break
        except:
            print("Error in initiating lidar, trying again..")
            lidar.disconnect()
            del lidar
            time.sleep(5)
            lidar = RPLidar(port_address)

    lidar.disconnect()
    del lidar
    lidar = RPLidar(port_address)
    # lidar.stop()
    # return lidar, scanIterator
    return lidar

def readLidarData(scanIterator):
    return next(scanIterator)

def stop2DLidar(lidar):
    lidar.stop()
    time.sleep(1)
    lidar.disconnect()
    return None

# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__ == '__main__':
    product_name = 'CP2102 USB to UART Bridge Controller'
    ports = serial.tools.list_ports.comports()
    port_address = ''
    for port in ports:
        if port.product==product_name:
            port_address = port.device
    if port_address=='':
        print("Device not found. Exiting.")
        sys.exit(1)
    print(f"Lidar found at port: {port_address}")
    # lidar = RPLidar(port_address)
    # time.sleep(5)
    # info = lidar.get_info()
    # print(info)
    #
    # health = lidar.get_health()
    # print(health)
    # for i, scan in enumerate(lidar.iter_scans()):
    #     print('%d: Got %d measures' % (i, len(scan)))
    #     if i > 10:
    #         ...
    #         break
    #     ...
    # lidar.stop()
    # lidar.stop_motor()
    # lidar.disconnect()


    # run(sys.argv[1])
    readAndParseDataRPLidar(port_address)
