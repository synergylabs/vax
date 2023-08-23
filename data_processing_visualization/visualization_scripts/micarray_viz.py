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
import matplotlib.pyplot as plt
import seaborn as sns


userscount= 7
users = [f'P{xr}' for xr in range(1,userscount+1)]
users = ['P3']

for user in users:
    print(f"Creating micarray visualizations for {user}")
    user_dir = f"/Volumes/Vax Storage/processed_data/{user}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    activity_dirs = glob.glob(f"{user_dir}/*")
    ID_IDX = 0
    X_IDX, Y_IDX, Z_IDX = 1, 2, 3
    ACT_IDX = 4
    colors = [(255, 0, 0), (0, 255, 0),
              (0, 0, 255), (255, 255, 255)]
    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            micarray_data_file = f"{instance_dir}/micarray.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(micarray_data_file):
                micarray_data = pickle.load(open(micarray_data_file,"rb"))
                fft_matrix = []
                if len(micarray_data) > 0:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/micarray_viz.avi', fourcc, 1, (400, 400), isColor=True)
                    for ts, data in micarray_data:
                        sst_dict = data['SST']
                        sources = []
                        for mic_src in sst_dict.keys():
                            if not (sst_dict[mic_src][ID_IDX] == 0):
                                sources.append(mic_src)
                        # print(data['Timestamp'])

                        # Get SSL sources for energy calculation
                        ssl_dict = data['SSL']
                        ssl_sources = []
                        for mic_src in ssl_dict.keys():
                            if not (ssl_dict[mic_src][-1] == 0.):
                                ssl_sources.append(mic_src)

                        window_dimension = 400
                        multipler = window_dimension // 4
                        cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
                        cv2_img = cv2.line(cv2_img, (0, window_dimension // 2),
                                           (window_dimension, window_dimension // 2),
                                           (255, 255, 255), 1)
                        # cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0),
                        #                    (window_dimension // 2, window_dimension),
                        #                    (255, 255, 255), 4)
                        cv2_img = cv2.circle(cv2_img, (window_dimension // 2, window_dimension // 2), 10, (255, 255, 255), 3)
                        for mic_src in sources:
                            idx, x, y, z, _ = sst_dict[mic_src]
                            px = min((window_dimension // 2) - int(y * multipler), window_dimension)
                            py = min((window_dimension // 2) - int(x * multipler), window_dimension)
                            z_size = int(4+(np.abs(z) * 10))
                            z_sign = np.sign(z)
                            # if z_sign < 0:
                            #     cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                            # else:
                            #     cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                            # print(px,py,z_size)

                        # Draw Circles for SSL Sources
                        for mic_src in ssl_sources:
                            x, y, z, e = ssl_dict[mic_src]
                            px = min((window_dimension // 2) - int(y * multipler), window_dimension)
                            py = min((window_dimension // 2) - int(x * multipler), window_dimension)
                            e = min(e, 1)
                            # point_color = (int(255 * e), 0, int(255 * (1 - e)))
                            point_color = (255, 255, 255)
                            z_size = int(np.abs(z) * 10)
                            z_sign = np.sign(z)
                            if z_sign < 0:
                                cv2_img = cv2.circle(cv2_img, (px, py), z_size, point_color, -1)
                            else:
                                cv2_img = cv2.circle(cv2_img, (px, py), z_size, point_color, -1)

                        img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
                        out.write(img_col)
                        fft_matrix.append(data['FFT'])
                    #     cv2.imshow('micarray', img_col)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                    # cv2.destroyAllWindows()
                    out.release()
                    fft_matrix = np.array(fft_matrix).T
                    fig,ax = plt.subplots(1,1,figsize=(fft_matrix.shape[1]*2,10))
                    sns.heatmap(fft_matrix,cbar=False)
                    plt.axis(False)
                    plt.savefig(f"{instance_viz_dir}/micarray_fft_viz.png",dpi=300,bbox_inches='tight')
                    plt.close()

                else:
                    print("micarray data doesn't exist...")
            else:
                print("micarray data doesn't exist...")
