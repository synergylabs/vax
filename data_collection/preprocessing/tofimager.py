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


def process_tofimager_data(raw_data_dir, df_labels, write_dir, prefix="tofimager"):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])
    # labels_data = [[]]*len(start_times)
    labels_data = []
    for _ in range(len(start_times)):
        labels_data.append(list())
    tofimager_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.b64"))
    ts = 0
    prev_label = ""
    for tofimager_file in tofimager_files:
        remainder = ""
        with open(tofimager_file, "r") as myFile:
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
                # tofimager_data.append(process_chunk(chunk))
                ts, data = process_chunk(chunk)
                if np.any((start_times < ts) & (end_times > ts)):
                    label_match_idxs = np.where((start_times < ts) & (end_times > ts))[0]
                    for label_match_idx in label_match_idxs:
                        if not (prev_label == labels[label_match_idx]['label_id']):
                            prev_label = labels[label_match_idx]['label_id']
                            print(f"Label Found ({prev_label}: {str(labels[label_match_idx])}")
                        labels_data[label_match_idx].append((ts, data))

    for idx, labels_info in enumerate(labels):
        labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        pickle.dump(labels_data[idx], open(f"{labels_dir}/tofimager.pb", "wb"))
        json.dump(labels_info, open(f"{labels_dir}/tofimager_label_info.json", "w"))
    return None



def visualize_tofimager_data(processed_data_dir):
    activity_dirs = glob.glob(f"{processed_data_dir}/*")
    for activity_dir in activity_dirs:
        instance_dirs =  glob.glob(f"{activity_dir}/*")
        for instance_dir in instance_dirs:
            print(f"Vizualizing instance: {instance_dir}")
            tofimager_data_file = f"{instance_dir}/tofimager.pb"
            instance_viz_dir = f"{instance_dir}/viz"
            if not os.path.exists(instance_viz_dir):
                os.makedirs(instance_viz_dir)
            if os.path.exists(tofimager_data_file):
                tofimager_data = pickle.load(open(tofimager_data_file,"rb"))
                if len(tofimager_data) > 0:
                    frame_size = (8,8)
                    scale_percent = 15000  # percent of original size
                    # initialize video writer
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out = cv2.VideoWriter(f'{instance_viz_dir}/tofimager_viz.avi', fourcc, 8, (int(frame_size[1]*(scale_percent//100)), int(frame_size[0]*(scale_percent//100))), isColor=True)
                    for ts, data in tofimager_data:
                        distance_data = data['distance'][0]
                        img = np.array(distance_data, dtype=np.float32).reshape(8, 8)
                        # remove any negative values and convert it into nan
                        img[img <= 0] = np.nan
                        img_mask = np.isnan(img)
                        # fill any nan values with average of all neighbouring pixels
                        img = cv2.inpaint(img, img_mask.astype(np.uint8), 3,
                                                      cv2.INPAINT_TELEA)
                        # replace left over nan values with average of neighbouring values
                        # nan_idxs = np.isnan(img)
                        # img[nan_idxs] = np.nanmean(img)
                        # rotate and flip image for better visualization
                        img = np.rot90(img, 1)
                        img = np.flip(img,0)
                        img = 255 * (img - img.min()) / (img.max() - img.min())
                        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                        # img_col = cv2.flip(img_col, 1)
                        # resizing image for better visualization
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_CUBIC)
                        out.write(img_resized)
                        # cv2.imshow('tofimager', img_col)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                    # cv2.destroyAllWindows()
                    out.release()
                else:
                    print("tofimager data doesn't exist...")
            else:
                print("tofimager data doesn't exist...")

    return None


def process_chunk(chunk):
    """
    Process one line of data from tofimager
    :param chunk:
    :return:
    """
    ts, encoded_data = chunk.split(" | ")
    ts = int(ts)
    data = pickle.loads(base64.b64decode(encoded_data.encode()))
    assert isinstance(data, dict)

    return (ts, data)
