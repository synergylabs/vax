'''
This file contains static config for video sensing.
Developer: Prasoon Patidar
Created at: 5th March, 2022
'''

class Config:
    VIDEO_SRC = 'logitech'
    RECORD = True

    # -----video config for oakdlite-----

    #video recording framerate
    fps = 10

    #video recording frame size
    video_width = 640
    video_height = 480

    # ----- config for depth/rgb recording -----

    # data collection folder
    data_folder = 'cache/video/logitech/'

    #max duration in single video file
    max_duration_per_video = 30

    #video codec
    video_codec = 'MJPG' # for OSX use 'MJPG', for linux 'XVID'
