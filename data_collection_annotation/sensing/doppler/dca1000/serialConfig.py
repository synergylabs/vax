"""
Main driver class for doppler data collection from awr1642boost-ods
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
import serial
from serial.tools import list_ports

# config parameters (not to be changed)
AWR_DEVICE_NAME = 'AWR1642BOOST-ODS'
# CLIPORT_ADDR = '/dev/ttyACM0'
CLIPORT_BAUDRATE = 115200
# DATAPORT_ADDR = '/dev/ttyACM1'
DATAPORT_BAUDRATE = 921600


def configure_serial(configFileName,cliport_address):
    cliport = serial.Serial(cliport_address, CLIPORT_BAUDRATE)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        cliport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)
    return cliport


def configure_dca(dcaBinaryPath, ldvsConfigFilePath):
    # Check status of board
    # response = subprocess.call([dcaBinaryPath, 'query_sys_status', ldvsConfigFilePath])
    # new_lib = '/home/vax/sensors/doppler/DCA1000/SourceCode/Release'
    # if 'LD_LIBRARY_PATH' not in os.environ.keys():
    #     os.environ['LD_LIBRARY_PATH'] =''
    # if not new_lib in os.environ['LD_LIBRARY_PATH']:
    #     os.environ['LD_LIBRARY_PATH'] += ':' + new_lib
    #     try:
    #         os.execv(sys.argv[0], sys.argv)
    #     except Exception as e:
    #         sys.exit('EXCEPTION: Failed to Execute under modified environment, ' + e)

    response = subprocess.run([dcaBinaryPath, 'query_sys_status', ldvsConfigFilePath], capture_output=True)
    success_response = "System is connected."
    if success_response in response.stdout.decode():
        print("DCA Board connected...")
    else:
        print("DCA Board disconnected. Exiting.")
        sys.exit(1)

    # Setup fpga and record setting with config
    response = subprocess.run([dcaBinaryPath, 'fpga', ldvsConfigFilePath], capture_output=True)
    success_response = 'FPGA Configuration command : Success'
    if success_response in response.stdout.decode():
        print("DCA FPGA Setup complete...")
    else:
        print("DCA FPGA setup error. Exiting.", response.stdout.decode())
        sys.exit(1)

    response = subprocess.run([dcaBinaryPath, 'record', ldvsConfigFilePath], capture_output=True)
    success_response = 'Configure Record command : Success'
    if success_response in response.stdout.decode():
        print("DCA LVDS Recording Setup complete...")
    else:
        print("DCA LVDS Recording setup error. Exiting.", response.stdout.decode())
        sys.exit(1)

    # response = subprocess.run([dcaBinaryPath, 'record', ldvsConfigFilePath], capture_output=True)
    # success_response = 'Configure Record command : Success'
    # if success_response in response.stdout.decode():
    #     print("DCA LVDS Recording Setup complete...")
    # else:
    #     print("DCA LVDS Recording setup error. Exiting.", response.stdout.decode())
    #     sys.exit(1)

    # recording_process = subprocess.Popen([dcaBinaryPath, 'start_record', ldvsConfigFilePath], stdout=subprocess.PIPE,
    #                                      stderr=subprocess.PIPE)
    # stdout, stderr = recording_process.communicate()
    # success_response = 'Start Record command : Success'
    # if success_response in stdout.decode():
    #     print("DCA Recording Started Successfully...")
    # else:
    #     print("DCA Recording start error. Exiting.", stdout.decode())
    #     sys.exit(1)
    # time.sleep(2)

    recording_process = subprocess.run([dcaBinaryPath, 'start_record', ldvsConfigFilePath],capture_output=True)
    # recording_process = subprocess.run(["/usr/bin/dbus-launch","/usr/bin/gnome-terminal","-x",
    #                                        "bash","-c",
    #                                     f"{dcaBinaryPath} start_record {ldvsConfigFilePath};exec bash"],capture_output=True)
    success_response = 'Start Record command : Success'
    if success_response in recording_process.stdout.decode():
        print("DCA Recording Started Successfully...")
    else:
        print("DCA Recording start error. Exiting.", recording_process.stdout.decode())
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
        print(f"DCA Recording Subprocess Spawned, PID: {dca_process_pid}")
        os.kill(dca_process_pid, signal.SIGKILL)
        print(f"DCA Recording Subprocess killed: {dca_process_pid}")
    return None


def stop_sensor(cliport_address):
    CLIport = serial.Serial(cliport_address, CLIPORT_BAUDRATE)

    CLIport.write(('sensorStop\n').encode())
    CLIport.close()


def reset_dca(dcaBinaryPath, ldvsConfigFilePath):
    response = subprocess.run([dcaBinaryPath, 'reset_fpga', ldvsConfigFilePath], capture_output=True)
    success_response = 'Reset FPGA command : Success'
    if success_response in response.stdout.decode():
        print("DCA Board Reset complete...")
    else:
        print("DCA Board Reset error. Exiting.", response.stdout.decode())
        sys.exit(1)
    return


if __name__=='__main__':
    filename = 'f30_hot.cfg'
    CLIPORT_HWID = 'USB VID:PID=0451:BEF3 SER=R0061036 LOCATION=1-4.4.4.4.2:1.0'
    CLIPORT_BAUDRATE = 115200
    # DATAPORT_ADDR = '/dev/ttyACM1'
    DATAPORT_HWID = 'USB VID:PID=0451:BEF3 SER=R0061036 LOCATION=1-4.4.4.4.2:1.3'
    DATAPORT_BAUDRATE = 921600
    ports = serial.tools.list_ports.comports()
    cliport_address = '/dev/ttyACM0'
    for port in ports:
        if CLIPORT_HWID in port.hwid:
            cliport_address = port.device
    print(f"Got CLI Port Address: {cliport_address}")
