'''
This is a wrapper to extract pose data from files collected with openpose backend for oak-d-lite
openpose = ['nose', 'neck', 'right shoulder', 'right elbow', 'right wrist',
        'left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee',
        'right ankle', 'left hip', 'left knee', 'left ankle', 'right eye',
        'left eye', 'right ear', 'left ear'
            ]
posenet=[
        'nose', 'left eye', 'right eye', 'left ear', 'right ear',
        'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
        'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
        'right knee', 'left ankle', 'right ankle'
    ]
'''

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


# Mapping openpose ids to posenet

openPose_to_poseNet = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
poseNet_skeleton_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9],
                   [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                   [13, 15], [12, 14], [14, 16]]

# get size of pose frame from openpose model config
MODELS_FOLDER = "/Users/ppatida2/VAX/vax/cache/oakdlite_models"
MODEL_NAME = 'openpose1'
def get_model_list():
    with open(os.path.join(MODELS_FOLDER, "models.json"), "r") as model_file:
        model_list = json.load(model_file)
    return model_list


pose_model_config = get_model_list()[MODEL_NAME]

# Folder path for pose files
user= 'phase2_user1'
collectionfolder = "Jan2023_user1"
dateofcollection = "28-10-2022"

write_dir = f"/Volumes/Vax Storage/processed_data/{user}"
if not os.path.exists(write_dir):
    os.makedirs(write_dir)


pose_files = glob.glob(f"/Volumes/Vax Storage/{collectionfolder}/{user}/*/pose_*.csv")
verify_data = False

def process_chunk(chunk):
    """
    Process one line of data from pose
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
        cv2.imshow("pose", img_col)
        if cv2.waitKey(1) == 27:
            print("Closing pose")
    assert isinstance(data, np.ndarray)

    return (ts, data)

pose_data = []
ts = 0
prev_label=  ""
for pose_file in pose_files:
    remainder = ""
    with open(pose_file, "r") as myFile:
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
            ts, data = process_chunk(chunk)
            print(pd.to_datetime(ts,unit='ns').tz_localize('UTC').tz_convert('US/Eastern'))
            pose_data.append((ts,data))
    print("\n\n")
    print("\n\n")
    print("\n\n")

print("Finished")
