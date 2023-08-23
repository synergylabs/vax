'''
This is main config file for audio sensing systems
'''


class Config:

    AUDIO_SRC = ''

    # max_duration_per_file = 30 # recording time per file(in mins)
    sampling_freq = 48000


    # --- Audio Config for Macbook Recording---
    macbok_mic_data_folder = 'cache/audio/macbook'
    audio_file_prefix = ''
    device_id = 4 # Very IMPORTANT: call python3 -m sounddevice and use corresponding device id for audio
    audio_channels = 2



