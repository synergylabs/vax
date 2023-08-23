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

for user in users:
    print(f"Creating mites visualizations for {user}")
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
            mites_data_file = f"{instance_dir}/mites.csv"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(mites_data_file):
                mites_data = pd.read_csv(mites_data_file,index_col=0)
                width = mites_data.shape[0] // 20
                height = 40
                fig, axn = plt.subplots(7, 1, sharex=True, gridspec_kw={'height_ratios': [6, 3, 3, 3, 3, 3, 1]})
                timestamps_data = mites_data['TimeStamp'].apply(lambda x: x.split(" ")[-1].split(".")[0])
                # grideye
                grideye_cols = [col for col in mites_data.columns if 'Gri' in col]
                df_grideye = pd.DataFrame(mites_data[grideye_cols].values.T,
                                          index=range(len(grideye_cols)),
                                          columns=timestamps_data)
                sns.heatmap(df_grideye, cbar=False, ax=axn[0], cmap=sns.color_palette("viridis"))
                axn[0].set_ylabel("grideye".upper(), fontsize=6)
                axn[0].set_yticks([])
                axn[0].set_xticks([])
                # accel_x
                accel_x_cols = [col for col in mites_data.columns if 'Accel-0' in col]
                df_accel_x = pd.DataFrame(mites_data[accel_x_cols].values.T,
                                          index=range(len(accel_x_cols)),
                                          columns=timestamps_data)
                sns.heatmap(df_accel_x, cbar=False, ax=axn[1], cmap=sns.color_palette("mako"))
                axn[1].set_ylabel("accel_x".upper(), fontsize=6)
                axn[1].set_yticks([])
                axn[1].set_xticks([])
                # accel_y
                accel_y_cols = [col for col in mites_data.columns if 'Accel-1' in col]
                df_accel_y = pd.DataFrame(mites_data[accel_y_cols].values.T,
                                          index=range(len(accel_y_cols)),
                                          columns=timestamps_data)
                sns.heatmap(df_accel_y, cbar=False, ax=axn[2], cmap=sns.color_palette("mako"))
                axn[2].set_ylabel("accel_y".upper(), fontsize=6)
                axn[2].set_yticks([])
                axn[2].set_xticks([])
                # accel_z
                accel_z_cols = [col for col in mites_data.columns if 'Accel-2' in col]
                df_accel_z = pd.DataFrame(mites_data[accel_z_cols].values.T,
                                          index=range(len(accel_z_cols)),
                                          columns=timestamps_data)
                sns.heatmap(df_accel_z, cbar=False, ax=axn[3], cmap=sns.color_palette("mako"))
                axn[3].set_ylabel("accel_z".upper(), fontsize=6)
                axn[3].set_yticks([])
                axn[3].set_xticks([])
                # mic
                mic_cols = [col for col in mites_data.columns if 'Mic' in col]
                df_mic = pd.DataFrame(mites_data[mic_cols].values.T,
                                      index=range(len(mic_cols)),
                                      columns=timestamps_data)
                sns.heatmap(df_mic, cbar=False, ax=axn[4], cmap=sns.color_palette("rocket"))
                axn[4].set_ylabel("mic".upper(), fontsize=6)
                axn[4].set_yticks([])
                axn[4].set_xticks([])
                # emi
                emi_cols = [col for col in mites_data.columns if 'EMI' in col]
                df_emi = pd.DataFrame(mites_data[emi_cols].values.T,
                                      index=range(len(emi_cols)),
                                      columns=timestamps_data)
                sns.heatmap(df_emi, cbar=False, ax=axn[5], cmap=sns.color_palette("rocket"))
                axn[5].set_ylabel("emi".upper(), fontsize=6)
                axn[5].set_yticks([])
                axn[5].set_xticks([])
                # mag
                mag_cols = [col for col in mites_data.columns if 'Mag' in col]
                df_mag = pd.DataFrame(mites_data[mag_cols].values.T,
                                      index=range(len(mag_cols)),
                                      columns=timestamps_data)
                sns.heatmap(df_mag, cbar=False, ax=axn[6], cmap=sns.color_palette("rocket"))
                axn[6].set_ylabel("mag".upper(), fontsize=6)
                axn[6].set_yticks([])
                plt.xticks(rotation=90,fontsize=6)
                plt.xlabel("")
                plt.savefig(f"{instance_viz_dir}/mites.png",dpi=300,bbox_inches='tight')
            else:
                print("mites data doesn't exist...")