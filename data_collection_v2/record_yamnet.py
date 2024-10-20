"""
Main driver class for  audio recording
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
import cv2
import numpy as np
import pandas as pd
import psutil
# from featurize.f1 import get_features
from database.DBO import DBO
from utils import get_logger
from deviceInterface import DeviceInterface
import sounddevice as sd
import soundfile as sf
import tflite_runtime.interpreter as tflite


# config parameters (not to be changed)
AUDIO_DEVICE_NAME = 'ReSpeaker 4 Mic Array'
ALLOWED_CHANNELS = [0]
YAMNET_MODEL_FILE = f'{Path(__file__).parent}/micarray/yamnet/yamnet.tflite'
YAMNET_CLASSES_FILE = f'{Path(__file__).parent}/micarray/yamnet/yamnet_class_map.csv'
MAX_DURATION_PER_FILE = 15 * 60
CHECKPOINT_FREQ = 20


def sigterm_handler(_signo, _stack_frame):
	# Raises SystemExit(0):
	sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)


class AudioReaderThread(threading.Thread):
	"""
	Reader thread for audio sensing from micarray device
	"""

	def __init__(self,
				 out_queue,
				 sampling_rate,
				 device_id,
				 num_channels,
				 logger,
				 feature_queue):
		threading.Thread.__init__(self)
		self.out_queue = out_queue
		self.logger = logger
		self.running = False
		self.sampling_rate = sampling_rate
		self.device_id = device_id
		self.num_channels = num_channels
		self.feature_queue = feature_queue

	def start(self):
		# connect with device for reading data
		...

		# mark this is running
		self.running = True
		self.logger.info(f"Starting reading data from {AUDIO_DEVICE_NAME} sensor...")
		# start
		super(AudioReaderThread, self).start()

	def stop(self):
		# destroy device relevant object
		...
		# set thread running to False
		self.running = False

	def callback(self, indata, frames, time, status):
		"""This is called (from a separate thread) for each audio block."""
		# if status:
		#     print(status, file=sys.stderr)
		self.out_queue.put(indata.copy())
		self.feature_queue.put(indata.copy())
		
	def run(self):
		try:
			curr_time = time.time()
			with sd.InputStream(samplerate=self.sampling_rate, device=self.device_id,
								channels=int(self.num_channels), callback=self.callback):
				while True:
					if not self.running:
						break
		except Exception as e:
			self.running = False
			self.logger.info("Exception thrown")
			traceback.print_exc(file=sys.stdout)
		finally:
			self.stop()




class AudioWriterThread(threading.Thread):
	"""
	Writer thread for doppler sensing from awr device
	"""

	def __init__(self, in_queue, write_dir, start_hour, end_hour, sampling_rate, num_channels,logger,prefix='yamnet', disk_limit=80):
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
		self.ckpt_file = '/tmp/yamnet.ckpt'
		self.num_ckpt_frames = 0
		self.sampling_rate = sampling_rate
		self.num_channels = num_channels


	def start(self):

		# mark this is running
		self.running = True
		self.logger.info(f"Starting writing data from {AUDIO_DEVICE_NAME} sensor...")
		# start
		super(AudioWriterThread, self).start()
		
	def renew_file(self):

		# release older csv
		self.out_file.close()

		# create new csv based on timestamp of next frame and reset current frame number
		time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		outfile_name = f'{self.write_dir}/{self.prefix}_{time_str}.wav'
		self.out_file = sf.SoundFile(outfile_name, mode='x', samplerate=self.sampling_rate,channels=int(self.num_channels), subtype='PCM_24')
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
			outfile_name = f'{self.write_dir}/{self.prefix}_{time_str}.wav'
			self.out_file = sf.SoundFile(outfile_name, mode='x', samplerate=self.sampling_rate,channels=int(self.num_channels), subtype='PCM_24')
			self.file_start_time = time.time()
			with open(self.ckpt_file,'w') as ckpt_f:
				ckpt_f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")},0.0')

			# run till thread is running
			while self.running:
				# run till this video exhausts
				while time.time() - self.file_start_time < MAX_DURATION_PER_FILE:
					if self.running:
						if self.in_queue.qsize() > 0:
							audio_frame = np.concatenate([self.in_queue.get() for _ in range(self.in_queue.qsize())])
							#ts = int(datetime.now().timestamp()*1e9)
							if (curr_hr >= self.start_hour) and (curr_hr <= self.end_hour) and (disk_frac<=self.disk_limit):
								self.out_file.write(audio_frame)
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



class AudioDevice(DeviceInterface):
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

		# initialize audioDevice
		self.name = AUDIO_DEVICE_NAME

		self.rpi_address = ''
		self.reader = None
		self.writer = None
		self.device_id = None
		self.num_channels = None
		self.sampling_rate = None
		return
	
	def is_available(self):
		"""
		Check if this particular device is available to collect data from
		Returns: True is device is available else False
		"""
		device_name = self.name
		print(f"Searching for device name: {device_name}")
		device_arr = sd.query_devices()
		selected_device= dict()
		device_id = -1
		for idx, device in enumerate(device_arr):
			if device_name in device['name']:
				selected_device = device
				device_id = idx
				break
		if device_id==-1:
			return False
		self.device_id = device_id
		self.num_channels = int(selected_device['max_input_channels'])
		self.sampling_rate= int(selected_device['default_samplerate'])
		print(f"Found device. Device Id: {self.device_id}, Channels: {self.num_channels}, Sampling Freq:{self.sampling_rate}")
		return True


	def startReader(self):
		"""
		Start reader thread, should not be called before initialization
		Returns: True if initialization successful, else false
		"""
		try:
			self.reader = AudioReaderThread(self.sensor_queue,
													self.sampling_rate,
													self.device_id,
													self.num_channels,
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
			self.writer = AudioWriterThread(self.sensor_queue,
													self.run_config['experiment_dir'],
													self.run_config['start_hour'],
													self.run_config['end_hour'],
													self.sampling_rate, 
													self.num_channels,
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
	logger = get_logger("yamnet", logdir = f'{Path(__file__).parent}/../../cache/logs',console_log=True)
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
		logger.error("remote end point not working properly...")
	#	sys.exit(1)
	# initialize device
	audioSensor = AudioDevice(run_config, sensor_queue, logger, feature_queue)

	# check if available
	if audioSensor.is_available():
		logger.info(f"- Found Sensor {audioSensor.name}-")
		audioSensor.startReader()
		audioSensor.startWriter()
	else:
		logger.error("Device not accessible, exiting...")
		sys.exit(1)
	# run for a given max duration
 
	# initialize tflite model
	yamnet_interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_FILE)
	yamnet_interpreter.allocate_tensors()
	input_details = yamnet_interpreter.get_input_details()
	output_details = yamnet_interpreter.get_output_details()
	input_shape = input_details[0]['shape']
	buffer_len = input_shape[1]
	logger.info(f"Loaded yamnet model with buffer len: {buffer_len}")

	# get yamnet classses
	yamnet_classes = pd.read_csv(YAMNET_CLASSES_FILE, index_col=0, header=0)['display_name'].values
	logger.info(f"Got yamnet class names: Shape: {yamnet_classes.shape}")

	try:
		num_frames = 0
		audio_frames = np.zeros((10,audioSensor.num_channels), dtype=np.float32)
		# ts_values = []
		# ts_features = []
		frame_data = {}
		start_ft_time = time.time()
		curr_timestamp = 0.
		while True:
			if feature_queue.qsize() > 0:
				audio_frame = np.concatenate([feature_queue.get() for _ in range(feature_queue.qsize())],axis=0)
				#print(frame_data.shape, audio_frame.shape)
				audio_frames = np.concatenate([audio_frames,audio_frame],axis=0)
				time_val_ = int(datetime.now().timestamp()*1e9)
				# Run yamnet interpreter
				if audio_frames.shape[0]>buffer_len:	
					yamnet_interpreter.set_tensor(input_details[0]['index'], audio_frames[-buffer_len:,0].reshape(1,-1))
					yamnet_interpreter.invoke()
					audio_output = yamnet_interpreter.get_tensor(output_details[0]['index'])[0]
					frame_ts  = int(time_val_//1e9) 
					if frame_ts not in frame_data:
						frame_data[frame_ts] = []
					frame_data[int(time_val_//1e9)].append(audio_output.flatten())
					# ts_values.append(int(time_val_//1e9)) # timestamp in milliseconds
					# ts_features.append(audio_output.flatten())
					audio_frames = audio_frames[-(buffer_len+ (buffer_len//100)):]
					num_frames += 1
				# Store output to database
				if time_val_ // 10**9 > curr_timestamp:
					curr_timestamp = time_val_ // 10**9
					if len(frame_data.keys()) > 0:
						try:
							ts_values = []
							ts_features = []
							for frame_ts in frame_data:
								ts_values.append(frame_ts)
								ts_median_val = np.nanmedian(frame_data[frame_ts], axis=0)
								ts_median_val = np.nan_to_num(ts_median_val)
								ts_features.append(ts_median_val)
							logger.info(f"got features: {len(ts_features)}, feature len: {len(ts_features[0])}")
							db_handler.write_features(run_config['name'], 'yamnet', run_config["featurizer"],
													np.array(ts_values), np.array(ts_features), logger)
						except KeyboardInterrupt:
							break
						except:
							logger.warning("Error in writing features to TSDB")
							logger.warning(traceback.format_exc())                                                      
					frame_data = {}
			if time.time() > start_ft_time + 10.:
				logger.info(f"Num audio Frames in last 10 Secs: {num_frames}")
				num_frames = 0
				start_ft_time = time.time()
		audioSensor.stopWriter()
		audioSensor.stopReader()
		# cv2.destroyWindow(window_name)
		logger.info(f"Stopped {audioSensor.name}")
	except KeyboardInterrupt:
		audioSensor.stopWriter()
		audioSensor.stopReader()
		logger.info(f"Stopped {audioSensor.name}")
	finally:
		audioSensor.stopWriter()
		audioSensor.stopReader()
		logger.info(f"Data Collection Complete {audioSensor.name}")
