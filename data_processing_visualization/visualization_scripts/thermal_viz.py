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


userscount= 7
users = [f'P{xr}' for xr in range(1,userscount+1)]

for user in users:
    print(f"Creating thermal visualizations for {user}")
    user_dir = f"/Volumes/Vax Storage/processed_data/{user}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    activity_dirs = glob.glob(f"{user_dir}/*")

    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            thermal_data_file = f"{instance_dir}/thermal.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(thermal_data_file):
                thermal_data = pickle.load(open(thermal_data_file,"rb"))
                if len(thermal_data) > 0:
                    frame_size = thermal_data[0][1].shape
                    # initialize video writer
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/thermal_viz.mp4', fourcc, 12, (frame_size[1], frame_size[0]), isColor=True)
                    for ts, data in thermal_data:
                        img = data.astype(np.float32)
                        img = 255 * (img - img.min()) / (img.max() - img.min())
                        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
                        out.write(img_col)
                        # cv2.imshow('thermal', img_col)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                    # cv2.destroyAllWindows()
                    out.release()
                else:
                    print("thermal data doesn't exist...")
            else:
                print("thermal data doesn't exist...")