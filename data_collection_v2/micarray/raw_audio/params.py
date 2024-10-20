"""
 Audio
"""
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.6
STFT_HOP_LENGTH_SECONDS = 0.03
EXAMPLE_WINDOW_SECONDS = 96 * 0.03   # Each example contains 96 10ms frames -- 2.88 sec
# EXAMPLE_HOP_SECONDS = 96*0.03*0.1    # with zero overlap.
EXAMPLE_HOP_SECONDS = 0.21    # modified from 0.2 to 0.21 on 2023/10/25

MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001  # Offset used for stabilized log of input mel-spectrogram.
NUM_MEL_BINS = 64  # Frequency bands in input mel-spectrogram patch.

"""
Motion
"""
WINDOW_LENGTH_IMU = 100
HOP_LENGTH_IMU = 10
