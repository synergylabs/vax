"""
This file contains preprocessing logic for micarray file
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

# EST_TZ = tz.gettz('America / NewYork')

user= 'P3'
collectionfolder = "Oct15"
dateofcollection = "15-10-2022"

write_dir = f"/Volumes/Vax Storage/processed_data/{user}"
if not os.path.exists(write_dir):
    os.makedirs(write_dir)


micarray_files = glob.glob(f"/Volumes/Vax Storage/{collectionfolder}/{user}/*/respeaker_*.csv")
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
#     cv2.namedWindow("micarray")
#     verify_data = True


def process_chunk(chunk):
    """
    Process one line of data from micarray
    :param chunk:
    :return:
    """
    ts, encoded_data = chunk.split(" | ")
    ts = int(ts)
    data = pickle.loads(base64.b64decode(encoded_data.encode()))
    if verify_data:
        det_matrix_vis = np.fft.fftshift(data, axes=1)
        img = det_matrix_vis.T
        img = img.astype(np.float32)
        img = 255 * (img - img.min()) / (img.max() - img.min())

        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imshow("micarray", img_col)
        if cv2.waitKey(1) == 27:
            print("Closing Doppler")
    assert isinstance(data, dict)

    return (ts, data)

# micarray_data = []
ts = 0
prev_label=  ""
for micarray_file in micarray_files:
    ts_offset = None
    file_ts_str = micarray_file.split("_")[-1].split(".")[0]
    file_ts_hr = datetime.strptime(file_ts_str, "%H%M%S")
    file_ts = pd.Series(datetime(year=user_date.year, month=user_date.month, day=user_date.day, hour=file_ts_hr.hour,
             minute=file_ts_hr.minute, second=file_ts_hr.second)).dt.tz_localize('America/New_York').astype(int).iloc[0]
    #Check first timestamp from file
    with open(micarray_file, "r") as myFile:
        first_line = myFile.readline()
        first_ts = int(first_line.split(" | ")[0])
        ts_offset = file_ts - first_ts


    remainder = ""
    with open(micarray_file, "r") as myFile:
        while True:
            chunk = [remainder]
            chunk_found = False
            while not chunk_found:
                try:
                    line = myFile.readline()
                    if line=="": # End of File
                        break
                except:
                    break
                if " ||" in line:
                    chunk_found = True
                    line, remainder = line.split(" ||")
                chunk.append(line)
            chunk = ''.join(chunk)
            if chunk == "":
                break
            # micarray_data.append(process_chunk(chunk))
            ts, data = process_chunk(chunk)
            ts = ts + ts_offset # offset to curb issue with rpi time sync
            if np.any((start_times<ts) & (end_times>ts)):
                label_match_idxs = np.where((start_times < ts) & (end_times > ts))[0]
                for label_match_idx in label_match_idxs:
                    if not (prev_label==labels[label_match_idx]['label_id']):
                        prev_label = labels[label_match_idx]['label_id']
                        print(f"Label Found ({prev_label}: {str(labels[label_match_idx])}")
                    labels_data[label_match_idx].append((ts, data))


for idx, labels_info in enumerate(labels):
    labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    pickle.dump(labels_data[idx], open(f"{labels_dir}/micarray.pb","wb"))
    json.dump(labels_info, open(f"{labels_dir}/micarray_label_info.json","w"))

print("Finished")
