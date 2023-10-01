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


def process_lidar2d_data(raw_data_dir, df_labels, write_dir, prefix="rplidar_2d"):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])
    # labels_data = [[]]*len(start_times)
    labels_data = []
    for _ in range(len(start_times)):
        labels_data.append(list())
    lidar2d_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.csv"))

    ts = 0
    prev_label = ""
    for lidar2d_file in lidar2d_files:
        remainder = ""
        with open(lidar2d_file, "r") as myFile:
            while True:
                chunk = [remainder]
                chunk_found = False
                while not chunk_found:
                    try:
                        line = myFile.readline()
                        if line == "":  # End of File
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
                # lidar2d_data.append(process_chunk(chunk))
                ts, data = process_chunk(chunk)
                if np.any((start_times < ts) & (end_times > ts)):
                    label_match_idxs = np.where((start_times < ts) & (end_times > ts))[0]
                    for label_match_idx in label_match_idxs:
                        if not (prev_label == labels[label_match_idx]['label_id']):
                            prev_label = labels[label_match_idx]['label_id']
                            print(f"Label Found for Lidar2D ({prev_label}: {str(labels[label_match_idx])}")
                        labels_data[label_match_idx].append((ts, data))

    for idx, labels_info in enumerate(labels):
        labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        pickle.dump(labels_data[idx], open(f"{labels_dir}/lidar2d.pb", "wb"))
        json.dump(labels_info, open(f"{labels_dir}/lidar2d_label_info.json", "w"))
    return None

def visualize_lidar2d_data(processed_data_dir):
    activity_dirs = glob.glob(f"{processed_data_dir}/*")
    for activity_dir in activity_dirs:
        instance_dirs = glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            lidar2d_data_file = f"{instance_dir}/lidar2d.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(lidar2d_data_file):
                lidar2d_data = pickle.load(open(lidar2d_data_file, "rb"))
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

    return None


def process_chunk(chunk):
    """
    Process one line of data from lidar2d
    :param chunk:
    :return:
    """
    ts, encoded_data = chunk.split(" | ")
    ts = int(ts)
    data = pickle.loads(base64.b64decode(encoded_data.encode()))
    assert isinstance(data, tuple)

    return (ts, data)
