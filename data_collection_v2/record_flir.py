#!/usr/bin/env python3
"""
Main driver class for  thermal recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
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
from flirpy.camera.lepton import Lepton
# custom libraries
from utils import get_logger, get_screen_size
from deviceInterface import DeviceInterface
import pickle
import base64
import signal
import psutil
import skimage.measure
import subprocess
from featurize.f1 import get_features
from database.DBO import DBO

# config parameters (not to be changed)
THERMAL_DEVICE_NAME = 'Flir'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 20

def sigterm_handler(_signo, _stack_frame):
	# Raises SystemExit(0):
	sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

class FlirReaderThread(threading.Thread):
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

	def start(self):

		# mark this is running
		self.running = True

		# start
		super(FlirReaderThread, self).start()

	def stop(self):
		# set thread running to False
		self.running = False

	def run(self):
		# connect with device for reading data
		try:
			with Lepton() as camera:
				while self.running:
					img = camera.grab()
					if img is not None:
						ts = time.time_ns()
						img = img.astype(np.float32)
						img = (img / 1e2) - 273
						self.out_queue.put((ts, img))
						if self.feature_queue is not None:
							self.feature_queue.put((ts, img))
		except Exception as e:
			self.running = False
			self.logger.info("Exception thrown")
			self.logger.info(traceback.format_exc())
		finally:
			self.stop()


class FlirWriterThread(threading.Thread):
	"""
	Writer thread for doppler sensing from awr device
	"""

	def __init__(self, in_queue, write_dir, start_hour, end_hour, logger,prefix='flir', disk_limit=80):
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
		self.ckpt_file = '/tmp/thermal.ckpt'
		self.num_ckpt_frames = 0


	def start(self):

		# mark this is running
		self.running = True
		self.logger.info(f"Starting writing data from {THERMAL_DEVICE_NAME} sensor...")
		# start
		super(FlirWriterThread, self).start()
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
						ts, img_data = self.in_queue.get()
						if (curr_hr >= self.start_hour) and (curr_hr <= self.end_hour) and (disk_frac<=self.disk_limit):
							# img_data = img_data.astype(np.float32)
							# img_data = (img_data / 1e2) - 273
							# todo: convert image from celcius to kelvin 100k
							encoded_data = base64.encodebytes(pickle.dumps(img_data)).decode()
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
			# with open(self.write_src, 'w') as f:
			#     while self.running:
			#         ts, img_data = self.in_queue.get()
			#         img_data = img_data.astype(np.float32)
			#         img_data = (img_data / 1e2) - 273
			#         # todo: convert image from celcius to kelvin 100k
			#         encoded_data = base64.encodebytes(pickle.dumps(img_data)).decode()
			#         f.write(f"{ts} | {encoded_data} ||")
			#         # f.write(f"{ts} | {str(img_data.tolist())}\n")
		except Exception as e:
			self.running = False
			self.logger.info("Exception thrown")
			traceback.print_exc(file=sys.stdout)


class FlirDevice(DeviceInterface):
	"""
	Device implementation for FlirDevice
	"""

	def __init__(self, run_config, sensor_queue, logger, feature_queue):
		"""
		Initialized required containers for sensing device
		Args:
			run_config: basic config for pipeline run
			logger: Logging object
		"""
		super().__init__(run_config, sensor_queue, logger, feature_queue)

		# initialize FlirDevice
		self.name = THERMAL_DEVICE_NAME
		self.reader = None
		self.writer = None
		return

	def is_available(self):
		"""
		Check if this particular device is available to collect data from
		Returns: True is device is available else False
		"""
		try:
			with Lepton() as camera:
				img = camera.grab()
			return True
		except:
			self.logger.error(f"Failed to detect device: {traceback.format_exc()}")
			return False

	def startReader(self):
		"""
		Start reader thread, should not be called before initialization
		Returns: True if initialization successful, else false
		"""
		try:
			self.reader = FlirReaderThread(self.sensor_queue,
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
			self.writer = FlirWriterThread(self.sensor_queue,
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
	logger = get_logger("flir", logdir = f'{Path(__file__).parent}/../../cache/logs', console_log=False)
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

	thermalSensor = FlirDevice(run_config, sensor_queue, logger, feature_queue)

	# check if available
	if thermalSensor.is_available():
		logger.info(f"- Found Sensor {thermalSensor.name}-")
		thermalSensor.startReader()
		thermalSensor.startWriter()
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
				# logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
				frame_time, thermal_frame = feature_queue.get()
				if frame_time // 10**9 > curr_timestamp:
					curr_timestamp = frame_time // 10**9
					if len(frame_data) > 0.:
						try:
							ts_values, features = get_features(frame_data,'flir')
							logger.info(f"got features: {ts_values.shape}:{ts_values}, {features.shape}")
							db_handler.write_features(run_config['name'], 'flir', run_config["featurizer"],
													  ts_values, features)
							# logger.info(f"original TS Values: {ts_values.shape}:{ts_values.astype(int)}, {features.shape}")
							# ts_values = ts_values*1000
							# logger.info(f"got features: {ts_values.shape}:{ts_values.astype(np.int64)}, {features.shape}")
							# db_handler.write_features(run_config['name'], 'flir', run_config["featurizer"],
							# 						  ts_values.astype(np.int64), features)
						except:
							logger.warning("Error in writing features to TSDB")
							logger.warning(traceback.format_exc())
					frame_data = []
				frame_data.append((frame_time, thermal_frame))
				num_frames += 1
			if time.time() > start_ft_time + 10.:
				logger.info(f"Num Thermal Frames in last 10 Secs: {num_frames}")
				num_frames = 0
				start_ft_time = time.time()

		thermalSensor.stopWriter()
		thermalSensor.stopReader()
		logger.info(f"Data Collection Complete {thermalSensor.name}")
	except KeyboardInterrupt:
		thermalSensor.stopWriter()
		thermalSensor.stopReader()
		logger.info(f"Stopped {thermalSensor.name}")
	except:
		logger.warning("Error in featurization code")
		logger.warning(traceback.format_exc())
		thermalSensor.stopWriter()
		thermalSensor.stopReader()
		logger.info(f"Stopped {thermalSensor.name}")
	finally:
		thermalSensor.stopWriter()
		thermalSensor.stopReader()
		logger.info(f"Stopped {thermalSensor.name}")

