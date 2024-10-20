#!/usr/bin/env python3
"""
Main driver class for  audio recording
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
import sys
import os
import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np
import librosa
import json
import pickle
import base64
import signal
import psutil

def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

# custom libraries
from sensing.utils import get_logger, get_screen_size
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
# AUDIO_DEVICE_NAME = 'HDA Intel PCH: ALC3202 Analog'
AUDIO_DEVICE_NAME = 'ReSpeaker 4 Mic Array'
CHECKPOINT_FREQ = 60

class ReaderThread(threading.Thread):
    """
    Reader thread for audio sensing from yeti device
    """

    def __init__(self,
                 out_queue,
                 write_src,
                 sampling_rate,
                 device_id,
                 num_channels,
                 logger,
                 viz_queue):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False
        self.sampling_rate = sampling_rate
        self.device_id = device_id
        self.num_channels = num_channels
        self.write_source = write_src
        self.viz_queue = viz_queue
        self.checkpoint = time.time()
        self.ckpt_file = '/tmp/audio.ckpt'

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting reading data from {AUDIO_DEVICE_NAME} sensor...")
        # start
        super(ReaderThread, self).start()

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

    def run(self):
        try:
            with sf.SoundFile(self.write_source, mode='x', samplerate=self.sampling_rate,
                              channels=int(self.num_channels), subtype='PCM_24') as file:
                with sd.InputStream(samplerate=self.sampling_rate, device=self.device_id,
                                    channels=int(self.num_channels), callback=self.callback):
                    while self.running:
                        next_packet = self.out_queue.get()
                        if next_packet is None:
                            self.running=False
                            continue
                        file.write(next_packet)
                        if time.time()-self.checkpoint>CHECKPOINT_FREQ:
                            with open(self.ckpt_file,'w') as ckpt_f:
                                ckpt_f.write(f'{datetime.now()}')
                            self.checkpoint = time.time()
                        # self.logger.info(f"Running read data from {AUDIO_DEVICE_NAME} sensor...")
                        if self.viz_queue is not None:
                            self.viz_queue.put(next_packet.copy())

        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Device(DeviceInterface):
    """
    Device implementation for Audio recording from
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue=None):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

        # initialize Device
        self.name = AUDIO_DEVICE_NAME
        self.device_id = -1
        self.num_channels = -1
        self.sampling_rate = -1
        self.reader = None
        self.writer = None
        self.visualizer = None

        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        self.logger.debug(f"Searching for device name: {self.name}")
        device_arr = sd.query_devices()
        selected_device = dict()
        for idx, device in enumerate(device_arr):
            if self.name in device['name']:
                selected_device = device
                self.device_id = idx
                break
        if self.device_id >= 0:
            self.num_channels = int(selected_device['max_input_channels'])
            self.sampling_rate = int(selected_device['default_samplerate'])
            self.logger.debug(
                f"Found audio device. Device Id: {self.device_id}, Channels: {self.num_channels}, Sampling Freq:{self.sampling_rate}")
            return True
        else:
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        write_source = None
        if write_source is None:
            time_str = datetime.now().strftime("%H%M%S")
            write_source = f"{self.run_config['experiment_dir']}/audio_{time_str}.wav"
        if self.device_id >= 0:
            self.reader = ReaderThread(self.sensor_queue,
                                       write_source,
                                       self.sampling_rate,
                                       self.device_id,
                                       self.num_channels,
                                       self.logger,
                                       viz_queue=self.viz_queue)
            if not self.reader.is_alive():
                self.reader.start()

            return True
        else:
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
        # write_source = None
        # if write_source is None:
        #     write_source = tempfile.mktemp(prefix='delme_rec_unlimited_',
        #                                    suffix='.wav', dir='')
        if self.device_id > 0:
            # self.writer = WriterThread(self.sensor_queue,
            #                                self.run_config['write_method'],
            #                                write_source,
            #                                self.sampling_rate,
            #                                self.num_channels,
            #                                self.logger)
            # self.writer.start()
            return True
        else:
            return False

    def stopWriter(self):
        """
        Gracefully stop writer thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            # self.writer.stop()
            # self.writer.join()
            return True
        except:
            self.logger.error(f"Failed to stop writer thread, {traceback.format_exc()}")
            return False


if __name__ == '__main__':
    logger = get_logger("vax_audio")
    default_config_file = 'config/dc_config.json'
    screen_width, screen_height = get_screen_size()
    visualize = True
    window_name = 'Audio FFt'
    try:
        config_file = sys.argv[1]
    except:
        config_file = default_config_file
        logger.warning(f"using default config file {default_config_file}")

    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins']*60
    t_data_collection_start = datetime.now()
    start_time = time.time()

    # Get mount point for harddisk dynamically
    harddisk_uuid = run_config['harddisk_uuid']
    rel_device_path = os.readlink(f'/dev/disk/by-uuid/{harddisk_uuid}')
    device_path_abs = os.path.normpath(os.path.join('/dev/disk/by-uuid',rel_device_path))
    partitions = psutil.disk_partitions()
    disk_mountpath = None 
    for partition in partitions:
        if partition.device==device_path_abs:
            disk_mountpath = partition.mountpoint

    if disk_mountpath is None:
        logger.error(f"Unable to find disk mountpoint for device UUID {harddisk_uuid}")
        sys.exit(1)
    logger.info(f"Found Mount point for disk, {disk_mountpath}")

    # get experiment dir
    experiment_dir = f"{disk_mountpath}/{run_config['name']}/{t_data_collection_start.strftime('%Y%m%d_%H')}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    run_config['experiment_dir'] = experiment_dir

    # initialize queues
    sensor_queue = Queue()
    if visualize:
        viz_queue = Queue()
    else:
        viz_queue = None

    # initialize device
    audioSensor = Device(run_config, sensor_queue, logger, viz_queue)

    # check if available
    if audioSensor.is_available():
        logger.info(f"- Found Sensor {audioSensor.name}-")
        audioSensor.startReader()
        audioSensor.startWriter()

    # run for a given max duration
    try:
        audio_frames = np.zeros((100, audioSensor.num_channels))

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, screen_width//2, 0)
        while (time.time() - start_time < max_duration):
            if visualize:
                if viz_queue.qsize() > 0:
                    # self.logger.info(f"Running viz data from {AUDIO_DEVICE_NAME} sensor...")
                    audio_frame = np.concatenate([viz_queue.get() for _ in range(viz_queue.qsize())])
                    audio_frames = np.concatenate([audio_frames, audio_frame])
                    audio_frames = audio_frames[-20000:]
                    S_fft = np.abs(librosa.stft(y=audio_frames.T, n_fft=256))
                    S_dB = librosa.amplitude_to_db(S_fft, ref=np.min).mean(axis=0)
                    img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow(window_name, img_col)
                    # if cv2.waitKey(1) == 27:
                    #     break  # esc to quit
                    if cv2.waitKey(1)==27:
                        break  # esc to quit
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

    except KeyboardInterrupt:
        audioSensor.stopWriter()
        audioSensor.stopReader()
        cv2.destroyWindow(window_name)
        logger.info(f"Stopped {audioSensor.name}")
    finally:
        sensor_queue.put(None)
        audioSensor.stopWriter()
        audioSensor.stopReader()
        logger.info(f"Data Collection Complete {audioSensor.name}")
