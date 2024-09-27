'''
This file contains static config for video sensing.
Developer: Prasoon Patidar
Created at: 5th March, 2022
'''

import depthai as dai

class Config:
    VIDEO_SRC = ''

    # -----video config for oakdlite-----

    # If set (True), the ColorCamera is downscaled from 1080p to 720p.
    # Otherwise (False), the aligned depth is automatically upscaled to 1080p
    downscaleColor = True

    #video recording framerate
    fps = 30

    #monocamera resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

    # ----- config for depth/rgb recording -----

    # data collection folder
    depth_data_folder = 'cache/video/oakdlite/depth/'
    rgb_data_folder = 'cache/video/oakdlite/rgb/'
    pose_data_folder = 'cache/video/oakdlite/pose/'

    #max duration in single video file
    max_duration_per_video = 30

    #video codec
    video_codec = 'XVID' # for OSX use 'MJPG', for linux 'XVID'
