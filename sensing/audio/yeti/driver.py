"""
Main driver class for Yeti audio recording
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""
import datetime
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

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
YETI_DEVICE_NAME = 'HDA Intel PCH: ALC3202 Analog'


class YetiReaderThread(threading.Thread):
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

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting reading data from {YETI_DEVICE_NAME} sensor...")
        # start
        super(YetiReaderThread, self).start()

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
                        file.write(next_packet)
                        # self.logger.info(f"Running read data from {YETI_DEVICE_NAME} sensor...")
                        if self.viz_queue is not None:
                            self.viz_queue.put(next_packet.copy())

        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class YetiVisualizerThread(threading.Thread):
    def __init__(self, viz_queue, logger, window_name='Audio FFT'):
        threading.Thread.__init__(self)
        self.viz_queue = viz_queue
        self.logger = logger
        self.running = False
        self.window_name = window_name

    def start(self):
        self.running = True
        self.logger.info(f"Starting data visualization from {YETI_DEVICE_NAME} sensor...")
        if not super(YetiVisualizerThread, self).is_alive():
            super(YetiVisualizerThread, self).start()

    def stop(self):
        self.running = False

    def run(self) -> None:
        audio_frames = np.zeros((100, 2))
        cv2.namedWindow(self.window_name)
        # self.logger.info(f"Waiting for Running viz data from {YETI_DEVICE_NAME} sensor...")
        while True:
            if self.running:
                if self.viz_queue.qsize() > 0:
                    # self.logger.info(f"Running viz data from {YETI_DEVICE_NAME} sensor...")
                    audio_frame = np.concatenate([self.viz_queue.get() for _ in range(self.viz_queue.qsize())])
                    audio_frames = np.concatenate([audio_frames, audio_frame])
                    audio_frames = audio_frames[-10000:]
                    S_fft = np.abs(librosa.stft(y=audio_frames.T, n_fft=256))
                    S_dB = librosa.amplitude_to_db(S_fft, ref=np.min).mean(axis=0)
                    img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imshow(self.window_name, img_col)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            else:
                time.sleep(2)
        cv2.destroyWindow(self.window_name)


class YetiWriterThread(threading.Thread):
    """
    Writer thread for audio sensing from yeti device
    """

    def __init__(self, in_queue, write_method, write_src, sampling_rate, num_channels, logger):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.logger = logger
        self.running = False
        if write_method == 'file':
            self.soundfile = sf.SoundFile(write_src,
                                          mode='x',
                                          samplerate=sampling_rate,
                                          channels=int(num_channels),
                                          subtype='PCM_24')

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {YETI_DEVICE_NAME} sensor...")
        # start
        super(YetiWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False
        self.soundfile.close()

    def run(self):
        try:
            while self.running:
                self.soundfile.write(self.in_queue.get())
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class YetiDevice(DeviceInterface):
    """
    Device implementation for Audio recording from Yeti
    """

    def __init__(self, run_config, sensor_queue, logger, viz_queue=None):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger, viz_queue)

        # initialize YetiDevice
        self.name = YETI_DEVICE_NAME
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
                f"Found yeti device. Device Id: {self.device_id}, Channels: {self.num_channels}, Sampling Freq:{self.sampling_rate}")
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
            time_str = datetime.datetime.now().strftime("%H%M%S")
            write_source = f"{self.run_config['experiment_dir']}/audio_{time_str}.wav"
        if self.device_id >= 0:
            self.reader = YetiReaderThread(self.sensor_queue,
                                           write_source,
                                           self.sampling_rate,
                                           self.device_id,
                                           self.num_channels,
                                           self.logger,
                                           viz_queue=self.viz_queue)
            if not self.reader.is_alive():
                self.reader.start()
            if self.viz_queue is not None:
                if self.visualizer is None:
                    self.visualizer = YetiVisualizerThread(self.viz_queue,
                                                           self.logger)
                if not self.visualizer.is_alive():
                    self.visualizer.start()

            return True
        else:
            return False

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            if self.viz_queue is not None:
                self.visualizer.stop()
                # self.visualizer.join()
            time.sleep(1)
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
            # self.writer = YetiWriterThread(self.sensor_queue,
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
