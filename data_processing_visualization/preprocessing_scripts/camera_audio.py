"""
This creates visualization and store it for instance level data
"""
import pandas as pd
import numpy as np
import base64
import pickle
import seaborn
import glob
from datetime import datetime, timedelta
import time
from dateutil import parser
import cv2
from dateutil import tz
import os
import sys
import json
import moviepy.editor as mp


userscount= 7
users = [f'P{xr}' for xr in range(10,11)]
for user in users:
    print(f"Creating camera audio data for {user}")
    user_dir = f"/Volumes/Vax Storage/processed_data/{user}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    activity_dirs = glob.glob(f"{user_dir}/*")

    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            camera_data_file = f"{instance_dir}/camera.mp4"
            audio_data_out_file = f"{instance_dir}/camera.wav"
            if os.path.exists(camera_data_file):
                camera_clip = mp.VideoFileClip(camera_data_file)
                camera_clip.audio.write_audiofile(audio_data_out_file)
            else:
                print("camera data doesn't exist...")