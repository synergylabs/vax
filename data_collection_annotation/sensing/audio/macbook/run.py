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
from sensing.audio.macbook.config import Config


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


if __name__ == '__main__':

    # initialize logger
    logger = get_logger('macbook_audio_runner', 'cache/logs/audio/macbook_mic')
    logger.info("------------ New Macbook Recording Run ------------")

    # get audio filename
    curr_time = time.time_ns()
    audio_filename = f'{Config.macbok_mic_data_folder}/{Config.audio_file_prefix}_ch_{Config.audio_channels}_fs_{Config.sampling_freq}_ts_{curr_time}.wav'
    if not os.path.exists(Config.macbok_mic_data_folder):
        os.makedirs(Config.macbok_mic_data_folder)

    # initialize queue for callback functionality
    q = queue.Queue()
    try:
        if Config.sampling_freq is None:
            device_info = sd.query_devices(Config.device_id, 'input')
            # soundfile expects an int, sounddevice provides a float:
            Config.sampling_freq = int(device_info['default_samplerate'])
        if audio_filename is None:
            audio_filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                             suffix='.wav', dir='')

        # Make sure the file is opened before recording anything:
        with sf.SoundFile(audio_filename, mode='x', samplerate=Config.sampling_freq,
                          channels=int(Config.audio_channels), subtype='PCM_24') as file:
            with sd.InputStream(samplerate=Config.sampling_freq, device=Config.device_id,
                                channels=int(Config.audio_channels), callback=callback):
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
