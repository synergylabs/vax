'''
This contains main file to record audio from macbook mic
Developer: Prasoon Patidar
Created on: 7th Mar 2022
'''

# Basic Libraries
import time
import traceback
import tempfile
import queue
import sys
from pathlib import Path
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import cv2
import pandas as pd
from datetime import datetime, timedelta
import tflite_runtime.interpreter as tflite
import logging
from logging.handlers import WatchedFileHandler

# Custom Libraries

def get_logger(logname, logdir='cache/logs',console_log=True):
    # Initialize the logger

    logger_master = logging.getLogger(logname)
    logger_master.setLevel(logging.DEBUG)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ## Add core logger handler

    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(logdir + '/' + logname + '.log')
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    ## Add stdout logger handler
    if console_log:
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(console_formatter)
        logger_master.addHandler(console_log)

    # initialize main logger
    logger = logging.LoggerAdapter(logger_master, {})

    return logger




def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def get_device_info(device_name='ReSpeaker 4 Mic Array'):
    print(f"Searching for device name: {device_name}")
    device_arr = sd.query_devices()
    selected_device= dict()
    device_id = -1
    for idx, device in enumerate(device_arr):
        if device_name in device['name']:
            selected_device = device
            device_id = idx
            break
    num_channels = int(selected_device['max_input_channels'])
    sampling_rate= int(selected_device['default_samplerate'])
    print(f"Found device. Device Id: {device_id}, Channels: {num_channels}, Sampling Freq:{sampling_rate}")
    return device_id, num_channels, sampling_rate


if __name__ == '__main__':

    # initialize logger
    logger = get_logger('yamnet', '/tmp/logs/audio')
    logger.info("------------ New Audio Recording Run ------------")

    # get device related info
    device_id, num_channels, sampling_rate = get_device_info(device_name='ReSpeaker 4 Mic Array') # default to logitech usb audio device
    # device_id, num_channels, sampling_rate = get_device_info() # default to logitech usb audio device

    # get audio filename
    curr_time = time.time_ns()
    audio_filename = None
    buffer_len = 55200

    # initialize tflite model
    yamnet_interpreter = tflite.Interpreter(model_path='yamnet.tflite')
    yamnet_interpreter.allocate_tensors()
    input_details = yamnet_interpreter.get_input_details()
    output_details = yamnet_interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    buffer_len = input_shape[1]
    logger.info(f"Loaded yamnet model with buffer len: {buffer_len}")

    # get yamnet classses
    yamnet_classes = pd.read_csv("yamnet_class_map.csv", index_col=0, header=0)['display_name'].values
    logger.info(f"Got yamnet class names: Shape: {yamnet_classes.shape}")
    

    # initialize queue for callback functionality
    q = queue.Queue()
    try:
        if sampling_rate is None:
            device_info = sd.query_devices(device_id, 'input')
            # soundfile expects an int, sounddevice provides a float:
            sampling_rate = int(device_info['default_samplerate'])
        if audio_filename is None:
            audio_filename = tempfile.mktemp(prefix='/tmp/delme_rec_unlimited_',
                                             suffix='.wav', dir='')
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(audio_filename, mode='x', samplerate=sampling_rate,
                          channels=int(num_channels), subtype='PCM_24') as file:
            with sd.InputStream(samplerate=sampling_rate, device=device_id,
                                channels=int(num_channels), callback=callback):
                logger.info('#' * 80)
                logger.info('Create FFT Signal from Audio')
                logger.info('#' * 80)
                audio_frames = np.zeros((100,6), dtype=np.float32)
                num_frames = 0
                curr_time = time.time()
                # frame_idx = 0
                while True:
                    # audio_frame = q.get()
                    while(q.qsize()<=0):
                        ...
                    audio_frame = np.concatenate([q.get() for _ in range(q.qsize())])                    
                    file.write(audio_frame)
                    audio_frames = np.concatenate([audio_frames,audio_frame])
                    if audio_frames.shape[0]>buffer_len:
                        yamnet_interpreter.set_tensor(input_details[0]['index'], audio_frames[-buffer_len:,0].reshape(1,-1))
                        yamnet_interpreter.invoke()
                        audio_output = yamnet_interpreter.get_tensor(output_details[0]['index'])[0]
                        # get top 5 prediction classes from audio output
                        top_indexes = np.argsort(audio_output)[::-1][:5]
                        # get classnames and their probabilities
                        top_classes = yamnet_classes[top_indexes]
                        top_probs = audio_output[top_indexes]
                        # print probabilities
                        # logger.info(f"{datetime.now().strftime('%H:%M:%S')}: {list(zip(top_classes, top_probs))}")
                        num_frames+=1
                        audio_frames = audio_frames[-buffer_len:]
                        if time.time()-curr_time > 10:
                            logger.info(f"Frames in last 10 seconds: {num_frames}")
                            num_frames=0
                            curr_time = time.time()
                        # time.sleep(1)

    except KeyboardInterrupt:
        logger.info('\nRecording finished: ' + repr(audio_filename))
        logger.info("Exiting with status code: Success")
    except Exception as e:
        logger.info("Exception Occured")
        logger.info(traceback.print_exc())
        logger.info("Exiting with status code: Failure")
