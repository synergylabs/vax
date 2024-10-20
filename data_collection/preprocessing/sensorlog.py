"""
This file contains preprocessing logic for imu file
"""
import os.path

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
import pytz
import json
import matplotlib.pyplot as plt

def process_sensorlog_data(iwatch_base_folder, df_labels, write_dir,prefix="Apple Watch"):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])
    # labels_data = [[]]*len(start_times)
    labels_data = []
    for _ in range(len(start_times)):
        labels_data.append(list())
    useful_columns = ['loggingTime(txt)',
       'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)',
       'accelerometerAccelerationZ(G)',
       'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)',
       'motionRotationRateX(rad/s)', 'motionRotationRateY(rad/s)',
       'motionRotationRateZ(rad/s)', 'motionUserAccelerationX(G)',
       'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)',
       'motionQuaternionX(R)',
       'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)',
       'motionGravityX(G)', 'motionGravityY(G)', 'motionGravityZ(G)']
    for sensor_log_file in glob.glob(f"{iwatch_base_folder}/*{prefix}*.csv"):
        # check if file is empty
        if os.stat(sensor_log_file).st_size == 0:
            continue
        df_sensorlog = pd.read_csv(sensor_log_file,sep=',',index_col=False)
        df_sensorlog = df_sensorlog[useful_columns]
        df_sensorlog['ts'] = df_sensorlog['loggingTime(txt)'].apply(to_epoch_ns)
        for idx in range(len(labels)):
            df_sensorlog_idx = df_sensorlog[(df_sensorlog.ts >= start_times[idx]) & (df_sensorlog.ts <= end_times[idx])]
            if df_sensorlog_idx.shape[0] > 0:
                print(f"Found label({labels[idx]['label_id']}): {labels[idx]}")
                labels_data[idx].append(df_sensorlog_idx)

    for idx, labels_info in enumerate(labels):
        labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        if len(labels_data[idx]) > 0:
            df_sensorlog_label = pd.concat(labels_data[idx], ignore_index=True).sort_values(by='ts')
            df_sensorlog_label.to_csv(f"{labels_dir}/sensorlog.csv")
            json.dump(labels_info, open(f"{labels_dir}/sensorlog_label_info.json", "w"))
    return

def to_epoch_ns(time_str):
    dt = datetime.fromisoformat(time_str)
    dt_utc = dt.astimezone(pytz.UTC)
    return int(dt_utc.timestamp() * 1e9)


def visualize_sensorlog_data(processed_data_dir):
    activity_dirs = glob.glob(f"{processed_data_dir}/*")
    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            sensorlog_data_file = f"{instance_dir}/sensorlog.csv"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(sensorlog_data_file):
                df_sensorlog = pd.read_csv(sensorlog_data_file,index_col=0)
                if df_sensorlog.shape[0] > 0:
                    df_sensorlog['ts'] = pd.to_datetime(df_sensorlog['ts'])
                    df_sensorlog = df_sensorlog.set_index('ts')
                    df_sensorlog = df_sensorlog.fillna(method='ffill')
                    df_sensorlog = df_sensorlog.fillna(method='bfill')
                    df_sensorlog['accel_mag'] = np.sqrt(df_sensorlog['accelerometerAccelerationX(G)']**2 + df_sensorlog['accelerometerAccelerationY(G)']**2 + df_sensorlog['accelerometerAccelerationZ(G)']**2)
                    df_sensorlog['rot_mag'] = np.sqrt(df_sensorlog['motionRotationRateX(rad/s)']**2 + df_sensorlog['motionRotationRateY(rad/s)']**2 + df_sensorlog['motionRotationRateZ(rad/s)']**2)
                    df_sensorlog['motion_mag'] = np.sqrt(df_sensorlog['motionYaw(rad)']**2 + df_sensorlog['motionPitch(rad)']**2 + df_sensorlog['motionRoll(rad)']**2)
                    df_sensorlog['gravity_mag'] = np.sqrt(df_sensorlog['motionGravityX(G)']**2 + df_sensorlog['motionGravityY(G)']**2 + df_sensorlog['motionGravityZ(G)']**2)
                    df_sensorlog['quat_mag'] = np.sqrt(df_sensorlog['motionQuaternionX(R)']**2 + df_sensorlog['motionQuaternionY(R)']**2 + df_sensorlog['motionQuaternionZ(R)']**2 + df_sensorlog['motionQuaternionZ(R)']**2)
                    # plot accel, rot, mag, gravity, quat
                    fig, ax = plt.subplots(5, 1, figsize=(20, 20))

                    df_sensorlog['accel_mag'].plot(ax=ax[0])
                    ax[0].set_title('Acceleration Magnitude')
                    df_sensorlog['rot_mag'].plot(ax=ax[1])
                    ax[1].set_title('Rotation Magnitude')
                    df_sensorlog['motion_mag'].plot(ax=ax[2])
                    ax[2].set_title('Motion Magnitude')
                    df_sensorlog['gravity_mag'].plot(ax=ax[3])
                    ax[3].set_title('Gravity Magnitude')
                    df_sensorlog['quat_mag'].plot(ax=ax[4])
                    ax[4].set_title('Quaternion Magnitude')

                    plt.savefig(f"{instance_viz_dir}/sensorlog.png")
                else:
                    print("sensorlog data doesn't exist...")
            else:
                print("sensorlog data doesn't exist...")
    return None
