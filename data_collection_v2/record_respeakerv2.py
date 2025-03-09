"""
Main driver class for  micarray recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""
# basic libraries
from datetime import datetime
from queue import Queue
import jstyleson as json
import subprocess
import socket
import threading
import time
import traceback
import sys
import os
from pathlib import Path
import pickle
import base64
import signal
import sounddevice as sd
import cv2
import numpy as np
import psutil
from featurize.f1 import get_features
from database.DBO import DBO


from utils import get_logger
from deviceInterface import DeviceInterface

# config parameters (not to be changed)
RESPEAKER_DEVICE_NAME = 'micarrayv2'
ODAS_EXECUTABLE_PATH = f'{Path(__file__).parent}/micarray/respeakerv2/odaslive'
ODAS_CONFIG_PATH = f'{Path(__file__).parent}/micarray/respeakerv2/devices/respeaker_usb_4_mic_array.cfg'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 20


def sigterm_handler(_signo, _stack_frame):
	# Raises SystemExit(0):
	sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


def get_device_id(device_name='ReSpeaker 4 Mic Array', logger=None):
	device_arr = sd.query_devices()
	if logger is not None:
		logger.info(device_arr)
	else:
		print(device_arr)
	for idx, device in enumerate(device_arr):
		if device_name in device['name']:
			hw_id = 2
			if 'hw:1' in device['name']:
				hw_id=1
			elif 'hw:3' in device['name']:
				hw_id=3
			if logger:
				logger.info(device)
			else:
				print(device)
			return hw_id
	if logger:
		logger.info(f"return default address")
	else:
		print(f"return default address")
	return 2



class Respeaker4MicReaderThread(threading.Thread):
	"""
	Reader thread for doppler sensing from awr1642boost
	"""

	def __init__(self, out_queue, logger, feature_queue):
		threading.Thread.__init__(self)
		self.out_queue = out_queue
		self.running = False
		self.logger = logger
		self.feature_queue = feature_queue

	def start(self):

		# mark this is running
		self.running = True

		# start
		super(Respeaker4MicReaderThread, self).start()

	def stop(self):
		# set thread running to False
		self.running = False

	def run(self):
		# connect with device for reading data
		micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],stdout=subprocess.PIPE)
		try:
			store1, store2 = "", ""
			sst_ssl_ctr = 1
			restart_count = 0
			num_frames_per_sec = 0
			curr_timestamp = time.time()
			while self.running:
				next_line = micarray_proc.stdout.readline().decode()[:-1]
				detObj_arr = []
				# self.logger.info(next_line)
				if len(next_line) == 0:
					self.logger.info("Empty line error")
					micarray_proc.kill()
					if restart_count > 2:
						self.logger.info(f"Too many restarts exiting process")
						time_val_ = time.time_ns()
						self.out_queue.put((time_val_, -1))
						self.feature_queue.put((0,time_val_, -1))
						break
					self.logger.info(f"restarting micarray process...")
					restart_count+=1
					micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],stdout=subprocess.PIPE)
					#break
				if sst_ssl_ctr == 1:
					store1 += next_line
					if next_line == '}':
						sst_ssl_ctr = 2
				elif sst_ssl_ctr == 2:
					store2 += next_line
					if next_line == '}':
						try:
							store1_arr = json.loads(store1)['src']
							store2_arr = json.loads(store2)['src']
						except:
							store1, store2, sst_ssl_ctr = "", "", 1
							continue
						num_frames_per_sec+=1
						if ("E" in store1_arr[0].keys()) & ("id" in store2_arr[0].keys()):
							detObj = {"SSL": store1_arr, "SST": store2_arr}
						elif ("id" in store1_arr[0].keys()) & ("E" in store2_arr[0].keys()):
							detObj = {"SSL": store2_arr, "SST": store1_arr}
						else:
							continue
						time_val_ = time.time_ns()
						self.out_queue.put((time_val_, detObj))
						detObj_arr.append((time_val_, detObj))
						store1, store2, sst_ssl_ctr = "", "", 1
						if time.time()-curr_timestamp > 0.1:
							if self.feature_queue is not None:
								self.feature_queue.put((num_frames_per_sec,time_val_, detObj_arr))
							#self.logger.info(f"Total frames per second: {num_frames_per_sec}")
							curr_timestamp = time.time()
							num_frames_per_sec = 0.
							detObj_arr= []
				else:
					self.logger.info(next_line)
					break
			self.logger.debug(f"{self.running}")
			self.logger.debug("Closing Reader Process")
			micarray_proc.kill()
		except Exception as e:
			self.running = False
			micarray_proc.kill()
			self.logger.info("Exception thrown")
			self.logger.info(traceback.format_exc())
		finally:
			self.stop()


class Respeaker4MicWriterThread(threading.Thread):
	"""
	Writer thread for doppler sensing from awr device
	"""

	def __init__(self, in_queue, write_dir, start_hour, end_hour, logger, prefix='micarrayv2', disk_limit=80):
		threading.Thread.__init__(self)

		self.in_queue = in_queue
		self.logger = logger
		self.write_dir = write_dir
		self.is_running = False
		self.prefix = prefix
		self.out_file = None
		self.disk_limit=disk_limit
		self.start_hour = start_hour
		self.end_hour = end_hour		
		self.file_start_time = None
		self.checkpoint = time.time()
		self.ckpt_file = '/tmp/micarray.ckpt'
		self.num_ckpt_frames = 0

	def start(self):
		# connect with device for reading data
		time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
		self.file_start_time = time.time()

		# mark this is running
		self.running = True
		self.logger.info(f"Starting writing data from {RESPEAKER_DEVICE_NAME} sensor...")
		# start
		super(Respeaker4MicWriterThread, self).start()

	def stop(self):
		# destroy device relevant object
		# set thread running to False
		self.running = False

	def renew_file(self):

		# release older csv
		self.out_file.close()

		# create new csv based on timestamp of next frame and reset current frame number
		time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.out_file = open(f'{self.write_dir}/{self.prefix}_{time_str}.b64', 'w')
		# self.csv_out = csv.writer(self.out_file)
		self.file_start_time = time.time()

	def run(self):
		is_header_set = False
		with open(self.ckpt_file,'w') as ckpt_f:
			ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},0.0')
		try:
			curr_hr = int(datetime.now().strftime("%H"))
			#check for disk fraction limit
			disk_cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
			disk_frac = subprocess.check_output(disk_cmd, shell=True).decode("utf-8")
			disk_frac = int(disk_frac.split("%")[0])			
			# run till thread is running
			while self.running:
				# run till this video exhausts
				while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
					if self.running:
						time_val_, data_dict = self.in_queue.get()
						if data_dict==-1:
							self.out_file.close()
							self.running=False
							break
						if (curr_hr >= self.start_hour) and (curr_hr <= self.end_hour) and (disk_frac<=self.disk_limit):
							encoded_data = base64.encodebytes(pickle.dumps(data_dict)).decode()
							self.out_file.write(f"{time_val_} | {encoded_data} ||")
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


class Respeaker4MicDevice(DeviceInterface):
	"""
	Device implementation for Respeaker4MicDevice
	"""

	def __init__(self, run_config, sensor_queue, logger, feature_queue):
		"""
		Initialized required containers for sensing device
		Args:
			run_config: basic config for pipeline run
			logger: Logging object
		"""
		super().__init__(run_config, sensor_queue, logger, feature_queue)

		# initialize Respeaker4MicDevice
		self.name = RESPEAKER_DEVICE_NAME

		self.rpi_address = ''
		self.reader = None
		self.writer = None
		return

	def is_available(self):
		"""
		Check if this particular device is available to collect data from
		Returns: True is device is available else False
		"""
		device_id = get_device_id(logger=self.logger)
		if int(device_id) not in [1,2,3,4]:
			self.logger.error("Device not found successfully, do a manual testing")
			return False
		global ODAS_CONFIG_PATH 
		ODAS_CONFIG_PATH = ODAS_CONFIG_PATH.replace(".cfg",f"_d{device_id}.cfg")
		logger.info(f"Config Path: {ODAS_CONFIG_PATH}")

		micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],stdout=subprocess.PIPE)
		time.sleep(1)
		if micarray_proc.poll() is None:
			micarray_proc.kill()
			return True
		else:
			return False

	def startReader(self):
		"""
		Start reader thread, should not be called before initialization
		Returns: True if initialization successful, else false
		"""
		try:
			self.reader = Respeaker4MicReaderThread(self.sensor_queue,
													self.logger,
													self.feature_queue)
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
			self.writer = Respeaker4MicWriterThread(self.sensor_queue,
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
	logger = get_logger("respeakerv2", logdir = f'{Path(__file__).parent}/../../cache/logs',console_log=True)
	default_config_file = f'{Path(__file__).parent}/config.json'
	try:
		config_file = sys.argv[1]
	except:
		config_file = default_config_file
		logger.warning(f"using default config file {default_config_file}")

	run_config = json.load(open(config_file, 'r'))
    

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
	# initialize device
	micarraySensor = Respeaker4MicDevice(run_config, sensor_queue, logger, feature_queue)

	# check if available
	if micarraySensor.is_available():
		logger.info(f"- Found Sensor {micarraySensor.name}-")
		micarraySensor.startReader()
		micarraySensor.startWriter()
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
				num_frames_server, time_val_, detObj_arr = feature_queue.get()
				if detObj_arr==-1:
					sys.exit(1)
				frame_data+=detObj_arr
				# print(num_frames_server, time_val_, detObj)
				if time_val_ // 10**9 > curr_timestamp:
					curr_timestamp = time_val_ // 10**9
					if len(frame_data) > 0.:
						try:
							ts_values, features = get_features(frame_data,'respeakerv2')
							#logger.info(f"got features: {ts_values.shape}, {features.shape}")
							db_handler.write_features(run_config['name'], 'respeakerv2', run_config["featurizer"],
													  ts_values, features)
							# ts_values = ts_values*1000
							# #logger.info(f"got features: {ts_values.shape}, {features.shape}")
							# db_handler.write_features(run_config['name'], 'respeakerv2', run_config["featurizer"],
							# 						  ts_values.astype(np.int64), features)
						except:
							logger.warning("Error in writing features to TSDB")
							logger.warning(traceback.format_exc())                                                      
						#post_thermal_features(frame_data)
					frame_data = []
				num_frames += num_frames_server
			if time.time() > start_ft_time + 10.:
				logger.info(f"Num Micarray Frames in last 10 Secs: {num_frames}")
				num_frames = 0
				start_ft_time = time.time()
		micarraySensor.stopWriter()
		micarraySensor.stopReader()
		# cv2.destroyWindow(window_name)
		logger.info(f"Stopped {micarraySensor.name}")
	except KeyboardInterrupt:
		micarraySensor.stopWriter()
		micarraySensor.stopReader()
		logger.info(f"Stopped {micarraySensor.name}")
	finally:
		micarraySensor.stopWriter()
		micarraySensor.stopReader()
		logger.info(f"Data Collection Complete {micarraySensor.name}")
