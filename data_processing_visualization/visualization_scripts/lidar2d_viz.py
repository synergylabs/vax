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
    print(f"Creating lidar2d visualizations for {user}")
    user_dir = f"/Volumes/Vax Storage/processed_data/{user}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    activity_dirs = glob.glob(f"{user_dir}/*")

    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            lidar2d_data_file = f"{instance_dir}/lidar2d.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(lidar2d_data_file):
                lidar2d_data = pickle.load(open(lidar2d_data_file,"rb"))
                if len(lidar2d_data) > 0:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/lidar2d_viz.avi', fourcc, 10, (400, 400), isColor=True)
                    for ts, data in lidar2d_data:
                        scan_x, scan_y = data
                        # print(scan_x[:10], scan_y[:10])
                        window_dimension = 400
                        divider = 8000 // window_dimension
                        cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
                        cv2_img = cv2.line(cv2_img, (0, window_dimension // 2),
                                           (window_dimension, window_dimension // 2),
                                           (255, 255, 255), 4)
                        cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0),
                                           (window_dimension // 2, window_dimension),
                                           (255, 255, 255), 4)
                        for x, y in zip(scan_x, scan_y):
                            if (np.abs(x) // divider < window_dimension) & (np.abs(y) // divider < window_dimension):
                                px = min((window_dimension // 2) + int(x // divider), window_dimension)
                                py = min((window_dimension // 2) + int(y // divider), window_dimension)
                                cv2_img = cv2.circle(cv2_img, (px, py), divider // 8, (255, 255, 255), -1)
                        img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_BONE)
                        out.write(img_col)
                    #     cv2.imshow('lidar2d', img_col)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                    # cv2.destroyAllWindows()
                    out.release()
                else:
                    print("lidar2d data doesn't exist...")
            else:
                print("lidar2d data doesn't exist...")
