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
sys.path.append(f"/home/vax/vax-rpi/sensing/")
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import librosa
import cv2

# Custom Libraries
from sensing.utils import get_logger
from sensing.micarray.raw_audio.vggish_input import wav_to_spectrogram 
#from sensing.audio.logitech.config import Config


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
    logger = get_logger('audio_viz', 'cache/logs/audio')
    logger.info("------------ New Audio Recording Run ------------")

    # get device related info
    device_id, num_channels, sampling_rate = get_device_info(device_name='ReSpeaker 4 Mic Array') # default to logitech usb audio device
    # device_id, num_channels, sampling_rate = get_device_info() # default to logitech usb audio device

    # get audio filename
    curr_time = time.time_ns()
    audio_filename = None
    buffer_len = 55200
    
    # initialize queue for callback functionality
    q = queue.Queue()
    try:
        if sampling_rate is None:
            device_info = sd.query_devices(device_id, 'input')
            # soundfile expects an int, sounddevice provides a float:
            sampling_rate = int(device_info['default_samplerate'])
        if audio_filename is None:
            audio_filename = tempfile.mktemp(prefix='cache/delme_rec_unlimited_',
                                             suffix='.wav', dir='')

        # Make sure the file is opened before recording anything:
        with sf.SoundFile(audio_filename, mode='x', samplerate=sampling_rate,
                          channels=int(num_channels), subtype='PCM_24') as file:
            with sd.InputStream(samplerate=sampling_rate, device=device_id,
                                channels=int(num_channels), callback=callback):
                logger.info('#' * 80)
                logger.info('Create FFT Signal from Audio')
                logger.info('#' * 80)
                audio_frames = np.zeros((100,6))
                log_mels = np.zeros((10,64))
                # frame_idx = 0
                while True:
                    # audio_frame = q.get()
                    while(q.qsize()<=0):
                        ...
                    audio_frame = np.concatenate([q.get() for _ in range(q.qsize())])                    
                    file.write(audio_frame)
                    audio_frames = np.concatenate([audio_frames,audio_frame])
                    if audio_frames.shape[0]>buffer_len:
                        log_mel = wav_to_spectrogram(audio_frames[-buffer_len:], sampling_rate)
                        log_mels = np.concatenate([log_mels,log_mel[-2:]])
                        print(time.time_ns(), audio_frame.shape, log_mel.shape, audio_frames.shape, log_mels.shape)          
                    log_mel_img = 255*(log_mels - np.min(log_mels))/(np.max(log_mels) - np.min(log_mels))
                    img_col = cv2.applyColorMap(log_mel_img.astype(np.uint8).transpose(), cv2.COLORMAP_JET)

                    # resizing image for better visualization
                    scale_percent = 300  # percent of original size
                    width = int(img_col.shape[1] * scale_percent / 100)
                    height = int(img_col.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow('Audio FFT', img_resized)
                    # if len(audio_frames)>=100:
                    audio_frames = audio_frames[-60000:]
                    log_mels = log_mels[-500:]
                    # frame_idx = 0.
                    # audio_frames = []
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit

    except KeyboardInterrupt:
        logger.info('\nRecording finished: ' + repr(audio_filename))
        logger.info("Exiting with status code: Success")
    except Exception as e:
        logger.info("Exception Occured")
        logger.info(traceback.print_exc())
        logger.info("Exiting with status code: Failure")
