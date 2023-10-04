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


def process_doppler_data(raw_data_dir, df_labels, write_dir, prefix="doppler"):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])
    # labels_data = [[]]*len(start_times)
    labels_data = []
    for _ in range(len(start_times)):
        labels_data.append(list())
    doppler_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.csv"))

    prev_label = ""
    for doppler_file in doppler_files:
        remainder = ""
        with open(doppler_file, "r") as myFile:
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
                ts, data = process_chunk(chunk)
                if np.any((start_times < ts) & (end_times > ts)):
                    label_match_idxs = np.where((start_times < ts) & (end_times > ts))[0]
                    for label_match_idx in label_match_idxs:
                        if not (prev_label == labels[label_match_idx]['label_id']):
                            prev_label = labels[label_match_idx]['label_id']
                            print(f"Label Found for Doppler ({prev_label}: {str(labels[label_match_idx])}")
                        labels_data[label_match_idx].append((ts, data))

    for idx, labels_info in enumerate(labels):
        labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        pickle.dump(labels_data[idx], open(f"{labels_dir}/doppler.pb", "wb"))
        json.dump(labels_info, open(f"{labels_dir}/doppler_label_info.json", "w"))
    return None

def visualize_doppler_data(processed_data_dir):
    activity_dirs = glob.glob(f"{processed_data_dir}/*")
    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            doppler_data_file = f"{instance_dir}/doppler.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(doppler_data_file):
                doppler_data = pickle.load(open(doppler_data_file,"rb"))
                if len(doppler_data) > 0:
                    frame_size = doppler_data[0][1]['det_matrix'].shape
                    # initialize video writer
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/doppler_viz.mp4', fourcc, 4, (frame_size[1], frame_size[0]), isColor=True)
                    for ts, data in doppler_data:
                        det_matrix_vis = np.fft.fftshift(data['det_matrix'], axes=1)
                        img = det_matrix_vis
                        img = img.astype(np.float32)
                        img = 255 * (img - img.min()) / (img.max() - img.min())
                        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                        out.write(img_col)
                        cv2.imshow('doppler', img_col)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    cv2.destroyAllWindows()
                    out.release()
                else:
                    print("doppler data doesn't exist...")
            else:
                print("doppler data doesn't exist...")
    return None

def process_chunk(chunk):
    """
    Process one line of data from doppler
    :param chunk:
    :return:
    """
    ts, encoded_data = chunk.split(" | ")
    ts = int(ts)
    data = pickle.loads(base64.b64decode(encoded_data.encode()))
    assert isinstance(data, dict)

    return (ts, data)

