#!/usr/bin/env python3
"""
This process kills all data collection processes gracefully(To be executed before shutdown)
"""
import time
from datetime import datetime
from dateutil import parser
import os
from config.kill_sensors_config import (sensors_ckpt_config)
from sensing.utils import get_logger, time_diff
import subprocess
import signal
import sys


# get pid for spawned child process
running_processes = subprocess.check_output(['ps', 'aux']).decode().split("\n")
for sensor_name in sensors_ckpt_config:
    # kill ongoing process ids for these sensors
    print(f"Terminating {sensor_name}...")
    sensor_proc_name = sensors_ckpt_config[sensor_name]['proc_file']
    sensor_processes = [xr for xr in running_processes if ((sensor_proc_name in xr) & ('exec bash' not in xr))]
    for sensor_process in sensor_processes:
        sensor_process_info = sensor_process.split(" ")[1:]
        sensor_process_pid = None
        for process_info_str in sensor_process_info:
            if not process_info_str == '':
                sensor_process_pid = int(process_info_str)
                break
        print(f"Sensor {sensor_name} process running, PID: {sensor_process_pid}")
        try:
            os.kill(sensor_process_pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

        time.sleep(2)
        print(f"Sensor Subprocess killed: {sensor_process_pid}")
    print(f"Successfully Terminated {sensor_name}...")

