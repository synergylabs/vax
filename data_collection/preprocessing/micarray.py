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


def process_micarray_data(raw_data_dir, df_labels, write_dir, prefix="micarrayv2"):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])
    # labels_data = [[]]*len(start_times)
    labels_data = []
    for _ in range(len(start_times)):
        labels_data.append(list())

    micarray_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.csv"))
    ts = 0
    prev_label = ""
    for micarray_file in micarray_files:
        remainder = ""
        with open(micarray_file, "r") as myFile:
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
                # micarray_data.append(process_chunk(chunk))
                ts, data = process_chunk(chunk)
                # ts = ts + ts_offset  # offset to curb issue with rpi time sync
                if np.any((start_times < ts) & (end_times > ts)):
                    label_match_idxs = np.where((start_times < ts) & (end_times > ts))[0]
                    for label_match_idx in label_match_idxs:
                        if not (prev_label == labels[label_match_idx]['label_id']):
                            prev_label = labels[label_match_idx]['label_id']
                            print(f"Label Found for Micarray ({prev_label}: {str(labels[label_match_idx])}")
                        labels_data[label_match_idx].append((ts, data))

    for idx, labels_info in enumerate(labels):
        labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        pickle.dump(labels_data[idx], open(f"{labels_dir}/micarray.pb", "wb"))
        json.dump(labels_info, open(f"{labels_dir}/micarray_label_info.json", "w"))
    return None


def visualize_micarray_data(processed_data_dir):
    activity_dirs = glob.glob(f"{processed_data_dir}/*")
    ID_IDX = 0
    X_IDX, Y_IDX, Z_IDX = 1, 2, 3
    ACT_IDX = 4
    colors = [(255, 0, 0), (0, 255, 0),
              (0, 0, 255), (255, 255, 255)]
    for activity_dir in activity_dirs:
        instance_dirs = glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            micarray_data_file = f"{instance_dir}/micarray.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(micarray_data_file):
                micarray_data = pickle.load(open(micarray_data_file, "rb"))
                if len(micarray_data) > 0:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/micarray_viz.avi', fourcc, 1, (400, 400), isColor=True)
                    for ts, data in micarray_data:
                        sst_dict = data['SST']
                        sources = []
                        for mic_src in range(len(sst_dict)):
                            if not (sst_dict[mic_src]['id'] == 0):
                                sources.append(mic_src)
                        # print(data['Timestamp'])

                        # Get SSL sources for energy calculation
                        ssl_dict = data['SSL']
                        ssl_sources = []
                        for mic_src in range(len(ssl_dict)):
                            if not (ssl_dict[mic_src]['E'] == 0.):
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
                        cv2_img = cv2.circle(cv2_img, (window_dimension // 2, window_dimension // 2), 10,
                                             (255, 255, 255), 3)
                        for mic_src in range(len(sources)):
                            idx, x, y, z = sst_dict[mic_src]['id'], sst_dict[mic_src]['x'], sst_dict[mic_src]['y'], \
                            sst_dict[mic_src]['z']
                            px = min((window_dimension // 2) - int(y * multipler), window_dimension)
                            py = min((window_dimension // 2) - int(x * multipler), window_dimension)
                            z_size = int(4 + (np.abs(z) * 10))
                            z_sign = np.sign(z)
                            # if z_sign < 0:
                            #     cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                            # else:
                            #     cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
                            # print(px,py,z_size)

                        # Draw Circles for SSL Sources
                        for mic_src in range(len(ssl_sources)):
                            x, y, z, e = ssl_dict[mic_src]['x'], ssl_dict[mic_src]['y'], ssl_dict[mic_src]['z'], \
                            ssl_dict[mic_src]['E']
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
                    #     cv2.imshow('micarray', img_col)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                    # cv2.destroyAllWindows()
                    out.release()

                else:
                    print("micarray data doesn't exist...")
            else:
                print("micarray data doesn't exist...")

    return None


def process_chunk(chunk):
    """
    Process one line of data from micarray
    :param chunk:
    :return:
    """
    ts, encoded_data = chunk.split(" | ")
    ts = int(ts)
    data = pickle.loads(base64.b64decode(encoded_data.encode()))
    assert isinstance(data, dict)

    return (ts, data)
