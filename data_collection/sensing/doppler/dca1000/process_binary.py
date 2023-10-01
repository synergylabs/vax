import mmwave as mm
from mmwave.dataloader import DCA1000, parse_raw_adc
import time
import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
import mmwave.dsp as dsp
import mmwave.clustering as clu
import numpy as np
import cv2
dca= DCA1000()

#parse_raw_adc('datacard_record_Raw_0.bin','datacard_record_Cleaned_0.bin')


numChirpsPerFrame = 32
numRxAntennas = 4
numTxAntennas = 2
numADCSamples = 256
fps = 20
IQ = 2
duration_in_sec = 60
total_frames = fps * duration_in_sec
#78641152, 78657536
raw_data = np.fromfile('datacard_record_Raw_0.bin', dtype=np.int16)
frame_length = numChirpsPerFrame * numRxAntennas * numADCSamples * 2
frames = np.stack([raw_data[xr * frame_length:(xr + 1) * frame_length] for xr in range(len(raw_data) // frame_length)])

cv2.namedWindow("doppler")
for frame_idx in range(frames.shape[0]):
    first_frame = frames[0,:]
    radar_cube = dca.organize(first_frame, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
    assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"
    fft1d_out = dsp.range_processing(radar_cube, window_type_1d=Window.BLACKMAN)
    det_matrix, aoa_input = dsp.doppler_processing(fft1d_out, num_tx_antennas=numTxAntennas, clutter_removal_enabled=True,
                                                   window_type_2d=Window.BLACKMAN,interleaved=False)
    img = det_matrix.T
    img = img.astype(np.float32)
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # resizing image for better visualization
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('doppler', img_resized)
    print(f"Frame Index: {frame_idx}")
    # time.sleep(0.5)
    if cv2.waitKey(1) == 27:
        print("Closing Doppler")

print("Finished")
