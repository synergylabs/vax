
#!/usr/bin/env python3
"""
Main driver class for  tofimager recording
Author: Prasoon Patidar
Created at: 12th Feb 2024
"""
from datetime import datetime
# basic libraries
import threading
import queue
import logging
import traceback
import sys
import time
import traceback
import tempfile
from queue import Queue
from pathlib import Path
import sys
import os
import cv2
import numpy as np
import jstyleson as json
from copy import copy
# custom libraries
from utils import get_logger, get_screen_size
from deviceInterface import DeviceInterface
import pickle
import base64
import signal
import psutil
import skimage.measure
from featurize.f1 import get_features
import subprocess
from database.DBO import DBO
import vl53l5cx_ctypes as vl53l5cx
# ~ from vl53l5cx.vl53l5cx import VL53L5CX
# ~ from vl53l5cx.api import VL53L5CX_RESOLUTION_8X8, VL53L5CX_RANGING_MODE_AUTONOMOUS,VL53L5CX_POWER_MODE_WAKEUP, VL53L5CX_TARGET_ORDER_CLOSEST


# config parameters (not to be changed)
LIDAR_DEVICE_NAME = 'Tofimager'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 20

def sigterm_handler(_signo, _stack_frame):
	# Raises SystemExit(0):
	sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

class TofimagerReaderThread(threading.Thread):
	"""
	Reader thread for doppler sensing from awr1642boost
	"""

	def __init__(self, out_queue, logger, feature_queue):
		threading.Thread.__init__(self)
		self.out_queue = out_queue
		self.logger = logger
		self.running = False
		self.camera = None
		self.feature_queue = feature_queue
		# old library code
		# ~ self.driver = VL53L5CX(nb_target_per_zone=1, disable_ambient_per_spad=True, disable_nb_spads_enabled=True,disable_signal_per_spad=True, disable_motion_indicator=True)
		# new library code
		self.logger.info("Uploading firmware, please wait...")
		self.driver = vl53l5cx.VL53L5CX()
		self.logger.info("Uploading firmware, ccompleted...")

	def start(self):
		# mark this is running
		self.running = True
		# start
		
		# old library code
		# ~ self.driver.init()
		# ~ self.driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
		# ~ self.driver.set_power_mode(VL53L5CX_POWER_MODE_WAKEUP)
		# ~ self.driver.set_ranging_frequency_hz(15)
		# ~ self.driver.set_ranging_mode(VL53L5CX_POWER_MODE_WAKEUP)
		# ~ self.driver.set_target_order(VL53L5CX_TARGET_ORDER_CLOSEST)
		# ~ self.driver.set_sharpener_percent(50)
		
		# new library code
		self.driver.set_resolution(8 * 8)
		self.driver.set_power_mode('Wakeup')
		# Enable motion indication at 8x8 resolution
		self.driver.enable_motion_indicator(8 * 8)
		# set motion capture range from 1m to 2m
		self.driver.set_motion_distance(1000, 2000)
		self.driver.set_ranging_frequency_hz(15)
		self.driver.set_integration_time_ms(5)
		self.logger.info("Driver initialized") 
		super(TofimagerReaderThread, self).start()

	def stop(self):
		# set thread running to False
		self.running = False

	def run(self):
		# connect with device for reading data
		try:
			# initialize driver and setup configs
			self.driver.start_ranging()
			self.logger.info("Ranging Started")
			while True:
				# ~ if self.driver.check_data_ready(): # old code
				if self.driver.data_ready():
					try:
						# ~ ranging_data = self.driver.get_ranging_data() # old code
						ranging_data = self.driver.get_data()
					except:
						logger.warning("Error in ranging data, skipping")
						logger.warning(traceback.format_exc())
					#self.logger.info(f"{num_frames}, {len(ranging_data.distance_mm)}, {len(ranging_data.range_sigma_mm)}, {len(ranging_data.reflectance)}")
					ts = time.time_ns()
					self.out_queue.put((ts,ranging_data))
					if self.feature_queue is not None:
						self.feature_queue.put((ts, ranging_data))
		except Exception as e:
			self.running = False
			self.logger.info("Exception thrown")
			self.logger.info(traceback.format_exc())
		finally:
			self.stop()


class TofimagerWriterThread(threading.Thread):
	"""
	Writer thread for doppler sensing from awr device
	"""

	def __init__(self, in_queue, write_dir, start_hour, end_hour, logger,prefix='tofimager', disk_limit=80):
		threading.Thread.__init__(self)

		self.in_queue = in_queue
		self.logger = logger
		self.write_dir = write_dir
		self.running = False
		self.prefix = prefix
		self.out_file = None
		self.disk_limit=disk_limit
		self.start_hour = start_hour
		self.end_hour = end_hour		
		self.file_start_time = None
		self.checkpoint = time.time()
		self.ckpt_file = '/tmp/tofimager.ckpt'
		self.num_ckpt_frames = 0


	def start(self):

		# mark this is running
		self.running = True
		self.logger.info(f"Starting writing data from {LIDAR_DEVICE_NAME} sensor...")
		# start
		super(TofimagerWriterThread, self).start()
	def renew_file(self):

		# release older csv
		self.out_file.close()

		# create new csv based on timestamp of next frame and reset current frame number
		time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
		# self.csv_out = csv.writer(self.out_file)
		self.file_start_time = time.time()

	def stop(self):
		# destroy device relevant object
		# set thread running to False
		self.running = False

	def run(self):
		try:
			# connect with device for reading data
			time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
			curr_hr = int(datetime.now().strftime("%H"))
			#check for disk fraction limit
			disk_cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
			disk_frac = subprocess.check_output(disk_cmd, shell=True).decode("utf-8")
			disk_frac = int(disk_frac.split("%")[0])						
			self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
			self.file_start_time = time.time()
			with open(self.ckpt_file,'w') as ckpt_f:
				ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},0.0')

			# run till thread is running
			while self.running:
				# run till this video exhausts
				while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
					if self.running:
						ts, ranging_data = self.in_queue.get()
						if (curr_hr >= self.start_hour) and (curr_hr <= self.end_hour) and (disk_frac<=self.disk_limit):
							encoded_data = base64.encodebytes(pickle.dumps(ranging_data)).decode()
							self.out_file.write(f"{ts} | {encoded_data} ||")
							self.num_ckpt_frames+=1
						else:
							self.num_ckpt_frames=-CHECKPOINT_FREQ
							if disk_frac>self.disk_limit:
								self.num_ckpt_frames=-10*CHECKPOINT_FREQ
						if time.time()-self.checkpoint>CHECKPOINT_FREQ:
							with open(self.ckpt_file,'w') as ckpt_f:
								ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},{round(self.num_ckpt_frames/CHECKPOINT_FREQ,2)}')
							self.checkpoint = time.time()
							self.num_ckpt_frames=0.
					else:
						self.out_file.close()
						break
				if self.running:
					curr_hr = int(datetime.now().strftime("%H"))
					#check for disk fraction limit
					disk_cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
					disk_frac = subprocess.check_output(disk_cmd, shell=True).decode("utf-8")
					disk_frac = int(disk_frac.split("%")[0])					
					self.renew_file()
		except Exception as e:
			self.running = False
			self.logger.info("Exception thrown")
			traceback.print_exc(file=sys.stdout)


class TofimagerDevice(DeviceInterface):
	"""
	Device implementation for TofimagerDevice
	"""

	def __init__(self, run_config, sensor_queue, logger, feature_queue):
		"""
		Initialized required containers for sensing device
		Args:
			run_config: basic config for pipeline run
			logger: Logging object
		"""
		super().__init__(run_config, sensor_queue, logger, feature_queue)

		# initialize TofimagerDevice
		self.name = LIDAR_DEVICE_NAME
		self.reader = None
		self.writer = None
		return

	def is_available(self):
		"""
		Check if this particular device is available to collect data from
		Returns: True is device is available else False
		"""
		try:
			self.logger.info("Checking Device...")
			self.driver = vl53l5cx.VL53L5CX()
			self.logger.info("Device Available...")
			return True		
			# old code	
			# ~ driver = VL53L5CX(nb_target_per_zone=1, disable_ambient_per_spad=True, disable_nb_spads_enabled=True,disable_signal_per_spad=True)
			# ~ if driver.is_alive():
				# ~ return True
			# ~ return False
		except:
			self.logger.error(traceback.format_exc())
			return False

	def startReader(self):
		"""
		Start reader thread, should not be called before initialization
		Returns: True if initialization successful, else false
		"""
		try:
			self.reader = TofimagerReaderThread(self.sensor_queue,
										   self.logger, self.feature_queue)
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
			self.writer = TofimagerWriterThread(self.sensor_queue,
										   self.run_config['experiment_dir'],
										   self.run_config['start_hour'],
										   self.run_config['end_hour'],
										   self.logger,
										   disk_limit=self.run_config['disk_limit'])
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


if __name__ == '__main__':
	logger = get_logger("tofimager", logdir = f'{Path(__file__).parent}/../../cache/logs', console_log=False)
	default_config_file = f'{Path(__file__).parent}/config.json'
	try:
		config_file = sys.argv[1]
	except:
		config_file = default_config_file
		logger.warning(f"using default config file {default_config_file}")

	run_config = json.load(open(config_file, 'r'))
	
	# change config name to be rpi eth0 mac address
	eth0_mac_cmd = "ifconfig eth0 | grep ether | awk 'END { print $2;}'"
	mac_address = subprocess.check_output(eth0_mac_cmd,shell=True).decode('utf-8')
	run_config['name']=f"rpi{mac_address.replace(':','')}".replace('\n','').replace('$','')
	
	# get experiment dir
	experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
	if not os.path.exists(experiment_dir):
		os.makedirs(experiment_dir)
	run_config['experiment_dir'] = experiment_dir

	# initialize queues
	sensor_queue = Queue()
	feature_queue = Queue()
	#initialize DBHandler
	db_handler = DBO()
	if not db_handler.is_http_success(logger):
		logger.error("tsdb not working properly, exiting...")
		sys.exit(1)
		
	tofimagerSensor = TofimagerDevice(run_config, sensor_queue, logger, feature_queue)

	# check if available
	if tofimagerSensor.is_available():
		logger.info(f"- Found Sensor {tofimagerSensor.name}-")
		tofimagerSensor.startReader()
		tofimagerSensor.startWriter()
	else:
		logger.error("Device not accessible, exiting...")
		sys.exit(1)

	# run for a given max duration
	try:
		num_frames = 0
		frame_data = []
		start_ft_time = time.time()
		curr_timestamp = 0.
		while True:
			if feature_queue.qsize() > 0:
				frame_time, tofimager_frame = feature_queue.get()
				if frame_time // 10**9 > curr_timestamp:
					curr_timestamp = frame_time // 10**9
					if len(frame_data) > 0.:
						try:
							ts_values, features = get_features(frame_data,'tofimager')
							# logger.info(f"got features: {ts_values.shape}, {features.shape}")
							db_handler.write_features(run_config['name'], 'tofimager', run_config["featurizer"],
													  ts_values, features)
							# ts_values = ts_values*1000
							# # logger.info(f"got features: {ts_values.shape}, {features.shape}")
							# db_handler.write_features(run_config['name'], 'tofimager', run_config["featurizer"],
							# 						  ts_values.astype(np.int64), features)
						except:
							logger.warning("Error in writing features to TSDB")
							logger.warning(traceback.format_exc())                    
					frame_data = []
				frame_data.append((frame_time, tofimager_frame))
				num_frames += 1
			if time.time() > start_ft_time + 10.:
				logger.info(f"Num Tofimager Frames in last 10 Secs: {num_frames}")
				num_frames = 0
				start_ft_time = time.time()

		tofimagerSensor.stopWriter()
		tofimagerSensor.stopReader()
		logger.info(f"Data Collection Complete {tofimagerSensor.name}")
	except KeyboardInterrupt:
		tofimagerSensor.stopWriter()
		tofimagerSensor.stopReader()
		logger.info(f"Stopped {tofimagerSensor.name}")
	finally:
		tofimagerSensor.stopWriter()
		tofimagerSensor.stopReader()
		logger.info(f"Stopped {tofimagerSensor.name}")


