"""
This is supervisory process to monitor and relaunch data collection processes
"""
import time
from datetime import datetime
from dateutil import parser
import os
from blink1.blink1 import blink1
from sensing.privacy_sensors.statusled.config import sensors_ckpt_config
from sensing.utils import get_logger, time_diff
import subprocess
import signal
import sys
STATUS_CHECK_FREQ = 180
MAX_CKPT_DIFFERENCE = 120
MAX_RESTART_DIFFERENCE = 300
logger = get_logger("DATA_COLLECTION_STATUS")
logger.info("Initializing Status LED")
with blink1() as b1:
    b1.fade_to_color(0,'white')
    checkpoint_times = {}
    restart_times = {}
    b1.fade_to_color(100, 'white')
    time.sleep(1)

    logger.info("Initializing Sensor Checkpoints")
    # initialize all checkpoints to same time
    for sensor_name in sensors_ckpt_config:
        curr_time = datetime.now()
        checkpoint_times[sensor_name] = datetime.now()
        restart_times[sensor_name] =  datetime.now()
        ckpt_file = sensors_ckpt_config[sensor_name]['ckpt_file']
        with open(ckpt_file, 'w') as ckpt_f:
            ckpt_f.write(f'{curr_time}')

    logger.info(f"Waiting {MAX_CKPT_DIFFERENCE} secs for data collection process to bootstrap...")
    time.sleep(MAX_CKPT_DIFFERENCE)
    logger.info(f"Done Waiting...")

    while True:
        sensor_to_sw_restart = []
        sensor_to_hw_restart = []
        # replace with latest checkpoints from ckpt files
        for sensor_name in sensors_ckpt_config:
            ckpt_file = sensors_ckpt_config[sensor_name]['ckpt_file']
            sensor_latest_time = None
            with open(ckpt_file, 'r') as f:
                sensor_latest_time = parser.parse(f.read())
            checkpoint_times[sensor_name] = sensor_latest_time

        curr_time = datetime.now()
        for sensor_name in sensors_ckpt_config:
            sensor_latest_time = checkpoint_times[sensor_name]
            if time_diff(sensor_latest_time, curr_time) > MAX_CKPT_DIFFERENCE:
                sensor_to_sw_restart.append(sensor_name)

        if len(sensor_to_sw_restart) >0.:
            logger.info(f"Need to software start {';'.join(sensor_to_sw_restart)} sensors")
            b1.fade_to_color(100, 'yellow')
        else:
            b1.fade_to_color(100, 'green')

        # check if any checkpoints are delayed more than MAX differnce
        '''
        if checkpoint delayed, switch led to yellow, and restart the process
        if previous restart time less than max restart difference, turn led to red, and ask user to power cycle 
        '''
        # get pid for spawned child process
        running_processes = subprocess.check_output(['ps', 'aux']).decode().split("\n")

        for sensor_name in sensor_to_sw_restart:
            # kill ongoing process ids for these sensors
            sensor_proc_name = sensors_ckpt_config[sensor_name]['proc_file']
            sensor_processes = [xr for xr in running_processes if (sensor_proc_name in xr)]
            for sensor_process in sensor_processes:
                sensor_process_info = sensor_process.split(" ")[1:]
                sensor_process_pid = None
                for process_info_str in sensor_process_info:
                    if not process_info_str == '':
                        sensor_process_pid = int(process_info_str)
                        break
                print(f"Sensor {sensor_name} process Spawned, PID: {sensor_process_pid}")
                os.kill(sensor_process_pid, signal.SIGINT)
                time.sleep(2)
                print(f"Sensor Subprocess killed: {sensor_process_pid}")
            sensor_launch_name = sensors_ckpt_config[sensor_name]['launch_file']

            #check if restarted in max restart period
            if time_diff(restart_times[sensor_name],datetime.now()) < MAX_RESTART_DIFFERENCE:
                logger.error(f"Hardware error, {sensor_name} restarted in less than {MAX_RESTART_DIFFERENCE} secs ago")
                logger.error(f"Turn off rig and restart")
                b1.fade_to_color(100,'red')
                sys.exit(1)

            #launch new process for given sensor
            proc_id = subprocess.Popen([sensor_launch_name])
            logger.info(f"Launching new process for {sensor_name}: {proc_id.pid}")
            restart_times[sensor_name]=datetime.now()
            time.sleep(5)


        logger.info("Completed checkpoint cycle.")
        time.sleep(MAX_CKPT_DIFFERENCE)




