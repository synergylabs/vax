"""
This file contains preprocessing logic for doppler file
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
import jstyleson as json
from sensing.utils import get_logger
from preprocessing.doppler import process_doppler_data, visualize_doppler_data
from preprocessing.lidar2d import process_lidar2d_data, visualize_lidar2d_data
from preprocessing.micarray import process_micarray_data, visualize_micarray_data
from preprocessing.thermalcam import process_thermal_data, visualize_thermal_data
from preprocessing.sensorlog import process_sensorlog_data, visualize_sensorlog_data
from preprocessing.tofimager import process_tofimager_data, visualize_tofimager_data
from preprocessing.camera import process_camera_data

default_config_file = '/Users/ppatida2/VAX/vax-public/data_collection/config/data_collection_config.json'
visualize = True

try:
    config_file = sys.argv[1]
except:
    config_file = default_config_file
    print(f"using default config file {default_config_file}")

preprocess_config = json.load(open(config_file, 'r'))

labels_raw_data_dir = f'../../nas_mnt/phase3/labels/{preprocess_config["name"]}'
# watch_raw_data_dir = f'{preprocess_config["out_data_dir"]}/iwatch'
# x_raw_data_dir = f'{preprocess_config["out_data_dir"]}/x_data'
phone_raw_data_dir = f'../../nas_mnt/phase3/raw_data/iphone/{preprocess_config["name"]}'

write_dir = f'../../nas_mnt/phase3/processed_data/{preprocess_config["name"]}'
# if not os.path.exists(write_dir):
#     os.makedirs(write_dir)

preprocessed_labels_file = f'{write_dir}/labels.csv'
if not os.path.exists(preprocessed_labels_file):
    # organize labels

    label_filepath = glob.glob(f"{labels_raw_data_dir}/*_labels.txt")[0]

    df_labels = pd.read_csv(label_filepath)

    df_labels = df_labels.drop_duplicates()
    df_labels['start_timestamp'] = df_labels['start_time'].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S.%f"))
    df_labels['start_timestamp'] = df_labels['start_timestamp'].dt.tz_localize(preprocess_config['timezone']).astype(int)

    df_labels['end_timestamp'] = df_labels['end_time'].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S.%f"))
    df_labels['end_timestamp'] = df_labels['end_timestamp'].dt.tz_localize(preprocess_config['timezone']).astype(int)

    df_labels = df_labels.sort_values(by='start_timestamp').reset_index(drop=True)
    df_labels['label_id'] = df_labels.index.astype(str)

    user = labels_raw_data_dir.split("/")[-1]
    df_labels['label_id'] = df_labels['label_id'].apply(lambda x: user + '_' + x.zfill(3))
    df_labels.to_csv(preprocessed_labels_file, index=False, sep="|")
else:
    df_labels = pd.read_csv(preprocessed_labels_file, sep="|")
# preprocess data across all modalities

# process_doppler_data(x_raw_data_dir, df_labels, write_dir)
# process_sensorlog_data(watch_raw_data_dir, df_labels, write_dir)
# process_tofimager_data(x_raw_data_dir, df_labels, write_dir)
process_camera_data(phone_raw_data_dir, df_labels, write_dir)
# process_lidar2d_data(x_raw_data_dir, df_labels, write_dir)
# process_micarray_data(x_raw_data_dir, df_labels, write_dir)
# process_thermal_data(x_raw_data_dir, df_labels, write_dir)


# store visualizations
# visualize_doppler_data(write_dir)
# visualize_lidar2d_data(write_dir)
# visualize_micarray_data(write_dir)
# visualize_thermal_data(write_dir)
# visualize_tofimager_data(write_dir)
# visualize_sensorlog_data(write_dir)