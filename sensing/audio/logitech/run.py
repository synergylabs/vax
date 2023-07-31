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

# Custom Libraries
from sensing.utils import get_logger
from sensing.audio.logitech.config import Config


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    # if status:
    #     print(status, file=sys.stderr)
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
    logger = get_logger('macbook_audio_runner', 'cache/logs/audio/macbook_mic')
    logger.info("------------ New Macbook Recording Run ------------")

    # get device related info
    # device_id, num_channels, sampling_rate = get_device_info(device_name='Yeti Stereo Microphone:') # default to logitech usb audio device
    device_id, num_channels, sampling_rate = get_device_info() # default to logitech usb audio device

    # get audio filename
    curr_time = time.time_ns()
    audio_filename = f'{Config.audio_data_folder}/{Config.audio_file_prefix}_ch_{num_channels}_fs_{sampling_rate}_ts_{curr_time}.wav'
    if not os.path.exists(Config.audio_data_folder):
        os.makedirs(Config.audio_data_folder)


    # initialize queue for callback functionality
    q = queue.Queue()
    try:
        if sampling_rate is None:
            device_info = sd.query_devices(device_id, 'input')
            # soundfile expects an int, sounddevice provides a float:
            sampling_rate = int(device_info['default_samplerate'])
        if audio_filename is None:
            audio_filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                             suffix='.wav', dir='')

        # Make sure the file is opened before recording anything:
        with sf.SoundFile(audio_filename, mode='x', samplerate=sampling_rate,
                          channels=int(num_channels), subtype='PCM_24') as file:
            with sd.InputStream(samplerate=sampling_rate, device=device_id,
                                channels=int(num_channels), callback=callback):
                logger.info('#' * 80)
                logger.info('press Ctrl+C to stop the recording')
                logger.info('#' * 80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        logger.info('\nRecording finished: ' + repr(audio_filename))
        logger.info("Exiting with status code: Success")
    except Exception as e:
        logger.info("Exception Occured")
        logger.info(traceback.print_exc())
        logger.info("Exiting with status code: Failure")
