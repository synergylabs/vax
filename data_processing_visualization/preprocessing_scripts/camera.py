"""
This file contains preprocessing logic for video file
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
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

# EST_TZ = tz.gettz('America / NewYork')

user = 'aigerim'
collectionfolder = "Oct15"
dateofcollection = "15-10-2022"

write_dir = f"/Volumes/Vax Storage/processed_data/{user}"
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

video_files = glob.glob(f"/Users/ppatida2/VAX/DataCollectionV1/{collectionfolder}/{user}/iphone/*.MOV")
df_video_files = pd.DataFrame(video_files, columns=['files'])
df_video_files['ts'] = df_video_files['files'].apply(lambda x: int(x.replace("_", "").split("/")[-1].split(".")[0]))
video_files = df_video_files.sort_values(by=['ts']).files.values.tolist()

label_files = glob.glob(f"../../../../DataCollectionV1/{collectionfolder}/{user}/{user}_labels*.txt")

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
df_labels['label_id'] = df_labels['label_id'].apply(lambda x: user + x.zfill(3))

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
#     cv2.namedWindow("video")
#     verify_data = True

def get_frame_ts(frame):
    frame_ts_str = pytesseract.image_to_string(frame[frame.shape[0] - 75:frame.shape[0] - 38, 480:800], lang='eng',
                                               config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:/')
    frame_ts = pd.Series(datetime.strptime(frame_ts_str[:-1], "%d/%m/%Y%H:%M:%S")).dt.tz_localize(
        "America/New_York").astype(int).iloc[0]
    return frame_ts


def binary_search_ts(video_frames, target_ts, epsilon_secs=2):
    low, high = 0, len(video_frames)
    while (high - low) > 2:
        mid = (low + high) // 2
        frame_mid = video_frames[mid]
        frame_ts = get_frame_ts(frame_mid)
        if np.abs(frame_ts - target_ts) / 1e9 <= epsilon_secs:
            return mid
        elif target_ts > frame_ts:
            low = mid
        else:
            high = mid
    return (high + low) // 2


samplerate = 0
ts = 0
prev_label = ""
count = 0
for video_file in video_files:
    video_frames = []
    vidcap = cv2.VideoCapture(video_file, )
    success, frame = vidcap.read()
    while success:
        video_frames.append(frame)
        success, frame = vidcap.read()
        count += 1
    video_ts_start = get_frame_ts(video_frames[0])
    video_ts_end = get_frame_ts(video_frames[-1])

    for idx, labels_info in enumerate(labels):
        label_start, label_end = start_times[idx], end_times[idx]
        if (label_start >= video_ts_start) & (label_end < video_ts_end):
            print(f"Found label({labels[idx]['label_id']}): {labels[idx]}")
            labels_dir = f"{write_dir}/{labels_info[' activity']}/{labels_info['label_id']}"
            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)
            out = cv2.VideoWriter(f"{labels_dir}/camera.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))
            label_start_idx = binary_search_ts(video_frames, label_start)
            label_end_idx = binary_search_ts(video_frames, label_end)
            for i in range(label_start_idx, label_end_idx + 1):
                out.write(video_frames[i])
            out.release()
            json.dump(labels_info, open(f"{labels_dir}/camera_label_info.json", "w"))
print("Finished")
