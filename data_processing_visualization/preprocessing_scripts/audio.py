"""
This file contains preprocessing logic for audio file
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
import json
import soundfile as sf

# EST_TZ = tz.gettz('America / NewYork')

user = 'P10'
collectionfolder = "Oct28_2"
dateofcollection = "28-10-2022"

write_dir = f"/Volumes/Vax Storage/processed_data/{user}"
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

audio_files = glob.glob(f"/Volumes/Vax Storage/{collectionfolder}/{user}/*/audio_*.wav")
label_files = glob.glob(f"/Volumes/Vax Storage/{collectionfolder}/{user}/{user}_labels.txt")
df_labels = None
for label_file in label_files:
    df_label_file = pd.read_csv(label_file)
    if df_labels is None:
        df_labels = df_label_file.copy(deep=True)
    else:
        df_labels = pd.concat([df_labels, df_label_file], ignore_index=True)

TIME_OFFSET_IN_SECS = 5
df_labels = df_labels.drop_duplicates()
user_date = datetime.strptime(dateofcollection, "%d-%m-%Y")
df_labels['start_timestamp'] = df_labels['start_time'].apply(lambda x: datetime.strptime(x.split("_")[-1], "%H%M%S.%f"))
df_labels['start_timestamp'] = df_labels['start_timestamp'].apply(
    lambda timehr: datetime(year=user_date.year, month=user_date.month, day=user_date.day, hour=timehr.hour,
                            minute=timehr.minute, second=timehr.second, microsecond=timehr.microsecond))
df_labels['start_timestamp'] = df_labels['start_timestamp'].dt.tz_localize("America/New_York").astype(int)

df_labels['end_timestamp'] = df_labels['end_time'].apply(lambda x: datetime.strptime(x.split("_")[-1], "%H%M%S.%f"))
df_labels['end_timestamp'] = df_labels['end_timestamp'].apply(
    lambda timehr: datetime(year=user_date.year, month=user_date.month, day=user_date.day, hour=timehr.hour,
                            minute=timehr.minute, second=timehr.second, microsecond=timehr.microsecond))
df_labels['end_timestamp'] = df_labels['end_timestamp'].dt.tz_localize("America/New_York").astype(int)

df_labels = df_labels.sort_values(by='start_timestamp').reset_index(drop=True)
df_labels['label_id'] = df_labels.index.astype(str)
df_labels['label_id'] = df_labels['label_id'].apply(lambda x: user + '_' + x.zfill(3))

labels = df_labels.to_dict('records')
ids = [xr['label_id'] for xr in labels]
start_times = np.array([xr['start_timestamp'] for xr in labels])
end_times = np.array([xr['end_timestamp'] for xr in labels])
# labels_data = [[]]*len(start_times)
labels_data = []
for _ in range(len(start_times)):
    labels_data.append(list())

verify_data = False
# if __name__ == '__main__':
#     cv2.namedWindow("audio")
#     verify_data = True

samplerate = 0
ts = 0
prev_label = ""
for audio_file in audio_files:
    ts_offset = None
    file_ts_str = audio_file.split("_")[-1].split(".")[0]
    file_ts_hr = datetime.strptime(file_ts_str, "%H%M%S")
    file_ts_start = \
        pd.Series(datetime(year=user_date.year, month=user_date.month, day=user_date.day, hour=file_ts_hr.hour,
                           minute=file_ts_hr.minute, second=file_ts_hr.second)).dt.tz_localize(
            'America/New_York').astype(int).iloc[0]

    audio_data, samplerate = sf.read(audio_file)
    file_ts_end = file_ts_start + int((audio_data.shape[0] / (samplerate)) * 1e9)
    for idx, labels_info in enumerate(labels):
        label_start, label_end = start_times[idx], end_times[idx]
        if (label_start >= file_ts_start) & (label_end < file_ts_end):
            print(f"Found label({labels[idx]['label_id']}): {labels[idx]}")
            label_start_offset = (label_start - file_ts_start) / 1e9
            label_end_offset = label_start_offset + (label_end - label_start) / 1e9
            label_start_idx = int(label_start_offset * samplerate)
            label_end_idx = int(label_end_offset * samplerate)
            labels_data[idx].append(audio_data[label_start_idx:label_end_idx,:])

for idx, labels_info in enumerate(labels):
    labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    if len(labels_data[idx]) > 0.:
        audio_data = np.concatenate(labels_data[idx],axis=0)
        sf.write(f"{labels_dir}/audio.wav",audio_data,samplerate=samplerate,subtype='PCM_24')
        json.dump(labels_info, open(f"{labels_dir}/audio_label_info.json", "w"))

print("Finished")
