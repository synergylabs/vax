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
from sensing.audio.logitech.config import Config


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def get_device_info(device_name='USB Device 0x46d:0x825'):
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
    device_id, num_channels, sampling_rate = get_device_info(device_name='HDA Intel PCH: ALC3202 Analog') # default to logitech usb audio device
    # device_id, num_channels, sampling_rate = get_device_info() # default to logitech usb audio device

    # get audio filename
    curr_time = time.time_ns()
    audio_filename = None


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
                audio_frames = np.zeros((100,2))
                # frame_idx = 0
                while True:
                    # audio_frame = q.get()
                    while(q.qsize()<=0):
                        ...
                    audio_frame = np.concatenate([q.get() for _ in range(q.qsize())])
                    file.write(audio_frame)
                    audio_frames = np.concatenate([audio_frames,audio_frame])
                    # frame_idx +=1
                    # if frame_idx>100:
                    # melSpectogram = librosa.feature.melspectrogram(y=np.concatenate(audio_frames).T, sr=sampling_rate, fmax=8000)
                    # S_dB = librosa.power_to_db(melSpectogram, ref=np.max).mean(axis=0)
                    # img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_RAINBOW)

                    S_fft = np.abs(librosa.stft(y=audio_frames.T, n_fft=256))
                    S_dB = librosa.amplitude_to_db(S_fft, ref=np.min).mean(axis=0)
                    img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_JET)

                    # resizing image for better visualization
                    # scale_percent = 300  # percent of original size
                    # width = int(img_col.shape[1] * scale_percent / 100)
                    # height = int(img_col.shape[0] * scale_percent / 100)
                    # dim = (width, height)
                    # img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                    cv2.imshow('Audio FFT', img_col)
                    # if len(audio_frames)>=100:
                    audio_frames = audio_frames[-40000:]
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
