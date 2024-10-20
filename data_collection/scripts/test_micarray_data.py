import _pickle
import binascii

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
import librosa


# micarray_file ="/Users/ppatida2/VAX/DataCollection/phase3/processed_data/P11/UsingMicrowave/P11_025/micarray.pb"
micarray_file ="/Users/ppatida2/VAX/DataCollection/phase3/processed_data/P12/UsingMicrowave/P12_011/micarray.pb"

micarray_data = pickle.load(open(micarray_file, "rb"))
micarray_frame_1 = micarray_data[0][1]

# extract activity level to get sound levels
activity_level = []
for ts, frame in micarray_data:
    activity_vals = [xr['activity'] for xr in frame['SST']]
    activity_level.append(np.max(activity_vals))
# plt.plot(activity_level)
# plt.show()
# is_activity = np.array(activity_level) > 0.99
# plt.plot(is_activity)
# plt.show()

# plot max E values in SSL
e_level = []
for ts, frame in micarray_data:
    e_vals = [xr['E'] for xr in frame['SSL']]
    e_level.append(np.max(e_vals))
plt.plot(e_level)
plt.show()

# find unique tag values in SST
tag_vals = []
for ts, frame in micarray_data:
    for sst_frame in frame['SST']:
        tag_vals.append(sst_frame['tag'])
tag_vals = np.unique(tag_vals)
print(tag_vals)

# find total unique ids for SST, and duration of each id (scatter plot with id on y and frame no. in x)
frame_idxs =[]
id_vals =[]
id_loc_x = []
id_loc_y = []
id_activity = []
id_energy = []
for frame_idx, (ts, frame) in enumerate(micarray_data):
    max_energy_frame = np.max([xr['E'] for xr in frame['SSL']])
    for sst_frame in frame['SST']:
        if sst_frame['tag'] == 'dynamic':
            frame_idxs.append(frame_idx)
            id_vals.append(sst_frame['id'])
            id_loc_x.append(sst_frame['x'])
            id_loc_y.append(sst_frame['y'])
            id_activity.append(sst_frame['activity'])
            id_energy.append(max_energy_frame)
df_sst = pd.DataFrame({'frame_idx': frame_idxs, 'id': id_vals, 'x': id_loc_x, 'y': id_loc_y, 'activity': id_activity, 'energy': id_energy})
df_sst = df_sst.sort_values(by=['id', 'frame_idx'])

# focus on only when activity is there (probability > 0.5
# df_sst = df_sst[df_sst['activity'] > 0.5]



seaborn.scatterplot(df_sst, x='x', y='y', hue='id')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

# plot x, y values for SSL sources, pick the one with maximum energy for each frame
frame_idxs = []
loc_x = []
loc_y = []
frame_energy = []

for frame_idx, (ts, frame) in enumerate(micarray_data[300:-300]):
    max_energy_frame = np.max([xr['E'] for xr in frame['SSL']])
    for ssl_frame in frame['SSL']:
        if ssl_frame['E'] == max_energy_frame:
            frame_idxs.append(frame_idx)
            loc_x.append(ssl_frame['x'])
            loc_y.append(ssl_frame['y'])
            frame_energy.append(max_energy_frame)

df_ssl = pd.DataFrame({'frame_idx': frame_idxs, 'x': loc_x, 'y': loc_y, 'energy': frame_energy})

df_ssl = df_ssl.sort_values(by='frame_idx')
seaborn.scatterplot(df_ssl[df_ssl.energy>df_ssl.energy.max()-0.01], x='x', y='y')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.title(df_ssl.energy.max())
plt.show()

# get max energy values, and create a spectrogram
frame_energy = []
for ts, frame in micarray_data:
    max_energy_frame = np.max([xr['E'] for xr in frame['SSL']])
    frame_energy.append(max_energy_frame)
frame_frequency = 120
frame_duration = len(micarray_data)/frame_frequency

amplitude = np.array(frame_energy)
time = np.linspace(0., frame_duration, len(amplitude))
plt.plot(time, amplitude)
plt.show()

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(amplitude)

# Convert the amplitude to decibels
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
from librosa import display
librosa.display.specshow(S_db, sr=frame_frequency, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# frame_spectogram = librosa.feature.melspectrogram(y=frame_energy, sr=frame_frequency)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(frame_spectogram, ref=np.max), y_axis='mel', x_axis='time')
print("finished")