import mmwave as mm
from mmwave.dataloader import DCA1000, parse_raw_adc
import time
import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
import mmwave.dsp as dsp
import mmwave.clustering as clu
import numpy as np
import cv2


def doppler_processing_custom(radar_cube, clutter_removal_enabled=True, interleaved=False,
                              window_type_2d=None, axis=1):
    '''
    Custom Doppler processing module for obtaining Doppler FFT on the N-D FFT data.

    Args:
        - radar_cube : raw data 3/4-D cube
    Output:
        - FFT across defined axis and in same shape and input
    Issues and To-do's:
        1. Interleaved implementation
    '''

    # assign doppler_axis as first axis (0). It's required for broadcasting and subsequent processing
    axes_vals = np.arange(len(radar_cube.shape))
    axes_vals[0] = axis
    axes_vals[axis] = 0
    fft1d_out = np.transpose(radar_cube, axes_vals)

    # Static Clutter Removal
    if clutter_removal_enabled:
        fft1d_out = dsp.compensation.clutter_removal(fft1d_out, axis=0)

    # Windowing 16x32
    if window_type_2d:
        fft2d_in = dsp.utils.windowing(fft1d_out, window_type_2d, axis=0)
    else:
        fft2d_in = fft1d_out

    # Perform FFT followed by FFTshift
    fft2d_out = np.fft.fft(fft2d_in, n=self.num_chirps * self.interp_factor, axis=0)
    fft2d_out = np.fft.fftshift(fft2d_out, axes=0)

    # transpose back the radarcube to original format before returning the output
    fft2d_out = np.transpose(fft2d_out, axes_vals)

    return fft2d_out


# parse_raw_adc('datacard_record_Raw_0.bin','datacard_record_Cleaned_0.bin')

## f20_2.cfg
numChirpsPerFrame = 64
numRxAntennas = 4
numTxAntennas = 2
numADCSamples = 256
fps = 20
## f30_hot.cfg
numChirpsPerFrame = 32
numRxAntennas = 4
numTxAntennas = 2
numADCSamples = 272
fps = 30

IQ = 2
global ADC_PARAMS
ADC_PARAMS = {'chirps': numChirpsPerFrame,  # 32
              'rx': numRxAntennas,
              'tx': numTxAntennas,
              'samples': numADCSamples / numTxAntennas,
              'IQ': 2,
              'bytes': 2}

dca = DCA1000()
cv2.namedWindow("doppler")
num_frames = 0
start_time = time.time()
while True:
    raw_adc_data = dca.read(timeout=.1)
    radar_cube = dca.organize(raw_adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas,
                              num_samples=numADCSamples)
    assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"
    fft1d_out = dsp.range_processing(radar_cube, window_type_1d=Window.HAMMING)
    fft1d_out = dsp.compensation.clutter_removal(fft1d_out)
    det_matrix, aoa_input = dsp.doppler_processing(fft1d_out, num_tx_antennas=numTxAntennas,
                                                   clutter_removal_enabled=True,
                                                   window_type_2d=Window.HAMMING, interleaved=False)
    # det_matrix
    img = det_matrix.T
    img = img.astype(np.float32)
    img = 255 * (img - img.min()) / (img.max() - img.min())

    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    # img_col = cv2.fastNlMeansDenoisingColored(img_col, None, 5, 5, 7, 21)
    # resizing image for better visualization
    scale_percent = 600  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('doppler', img_resized)
    num_frames += 1
    if time.time() > start_time + 10.:
        print(f"Num Frames in last 10 Secs: {num_frames}")
        num_frames = 0
        start_time = time.time()

    # print(f"Frame Index: {frame_idx}")
    # time.sleep(0.5)
    if cv2.waitKey(1) == 27:
        print("Closing Doppler")

print("Finished")
