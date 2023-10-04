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


default_config_file = 'config/data_collection_config.json'
visualize = True

try:
    config_file = sys.argv[1]
except:
    config_file = default_config_file
    print(f"using default config file {default_config_file}")

preprocess_config = json.load(open(config_file, 'r'))

user_raw_data_dir = f'{preprocess_config["out_data_dir"]}/{preprocess_config["name"]}'
date_folders = glob.glob(f"{user_raw_data_dir}/*/")

write_dir = f'{user_raw_data_dir}/processed_data'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# organize labels

label_filepath = glob.glob(f"{user_raw_data_dir}/*_labels.txt")[0]

df_labels = pd.read_csv(label_filepath)

df_labels = df_labels.drop_duplicates()
df_labels['start_timestamp'] = df_labels['start_time'].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S.%f"))
df_labels['start_timestamp'] = df_labels['start_timestamp'].dt.tz_localize(preprocess_config['timezone']).astype(int)

df_labels['end_timestamp'] = df_labels['end_time'].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S.%f"))
df_labels['end_timestamp'] = df_labels['end_timestamp'].dt.tz_localize(preprocess_config['timezone']).astype(int)

df_labels = df_labels.sort_values(by='start_timestamp').reset_index(drop=True)
df_labels['label_id'] = df_labels.index.astype(str)
user = user_raw_data_dir.split("/")[-1]
df_labels['label_id'] = df_labels['label_id'].apply(lambda x: user + '_' + x.zfill(3))

# preprocess data across all modalities

process_doppler_data(user_raw_data_dir, df_labels, write_dir)
process_lidar2d_data(user_raw_data_dir, df_labels, write_dir)
process_micarray_data(user_raw_data_dir, df_labels, write_dir)
process_thermal_data(user_raw_data_dir, df_labels, write_dir)

# store visualizations
visualize_doppler_data(write_dir)
visualize_lidar2d_data(write_dir)
visualize_micarray_data(write_dir)
visualize_thermal_data(write_dir)
