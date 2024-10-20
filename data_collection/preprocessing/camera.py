"""
This file contains preprocessing logic for video file
"""
import json
import os
import os.path
import subprocess
from datetime import datetime
import glob
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import moviepy.editor as mp


def get_frame_ts(frame):
    frame_ts_str = pytesseract.image_to_string(frame[frame.shape[0] - 75:frame.shape[0]-35, 478:800], lang='eng',
                                               config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:/.\ ')
    # frame_ts_str='14/10/2022 16:22:39'
    frame_ts = pd.Series(datetime.strptime(frame_ts_str[:-1], "%m/%d/%Y %H:%M:%S")).dt.tz_localize(
        "America/Los_Angeles").astype(int).iloc[0]
    return frame_ts


# def binary_search_ts(video_frames, target_ts, epsilon_secs=2):
#     low, high = 0, len(video_frames)
#     while (high - low) > 2:
#         mid = (low + high) // 2
#         frame_mid = video_frames[mid]
#         frame_ts = get_frame_ts(frame_mid)
#         if np.abs(frame_ts - target_ts) / 1e9 <= epsilon_secs:
#             return mid
#         elif target_ts > frame_ts:
#             low = mid
#         else:
#             high = mid
#     return (high + low) // 2

def process_camera_data(raw_data_dir, df_labels, write_dir, prefix=""):
    labels = df_labels.to_dict('records')
    ids = [xr['label_id'] for xr in labels]
    start_times = np.array([xr['start_timestamp'] for xr in labels])
    end_times = np.array([xr['end_timestamp'] for xr in labels])

    video_files = sorted(glob.glob(f"{raw_data_dir}/{prefix}*.mov"))
    for video_file in video_files:
        vidcap = cv2.VideoCapture(video_file, )
        success, frame = vidcap.read()
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_ts_start = get_frame_ts(frame)
        # video_ts_start_str = video_file.split("/")[-1].split(".")[0].split("_")[-2:]
        # video_ts_start_str = "_".join(video_ts_start_str)
        # video_ts_start = pd.Series(datetime.strptime(video_ts_start_str, "%Y%m%d_%H%M%S")).dt.tz_localize(
        #     "UTC").astype(int).iloc[0]
        video_ts_end = video_ts_start + int((frames * 1e9 / fps))
        for idx, labels_info in enumerate(labels):
            label_start, label_end = start_times[idx], end_times[idx]
            if (label_start >= video_ts_start) & (label_end < video_ts_end):
                print(f"Found label({labels[idx]['label_id']}): {labels[idx]}")
                labels_dir = f"{write_dir}/{labels_info[' activity'].strip()}/{labels_info['label_id']}"
                if not os.path.exists(labels_dir):
                    os.makedirs(labels_dir)
                outfile = f"{labels_dir}/camera.mp4"
                audiofile = f"{labels_dir}/camera.wav"
                if not os.path.exists(outfile):
                    label_start_duration = pd.to_datetime(label_start - video_ts_start, unit='ns').strftime('%H:%M:%S')
                    label_end_duration = pd.to_datetime(label_end - video_ts_start, unit='ns').strftime('%H:%M:%S')
                    _ = subprocess.run(
                        ['ffmpeg', '-ss', label_start_duration, '-to', label_end_duration, '-i', video_file, '-c', 'copy',
                         outfile], capture_output=True)

                    json.dump(labels_info, open(f"{labels_dir}/camera_label_info.json", "w"))
                if not os.path.exists(audiofile):
                    # get audio file for outfile
                    camera_clip = mp.VideoFileClip(outfile)
                    camera_clip.audio.write_audiofile(audiofile)
