# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

import numpy as np
from scipy.io import wavfile

from micarray.raw_audio import params
from micarray.raw_audio import mel_features


def wav_to_spectrogram(wav_data, sampling_rate):
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(wav_data,
                                               audio_sample_rate=sampling_rate,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=params.MEL_MIN_HZ,
                                               upper_edge_hertz=params.MEL_MAX_HZ)
    # (16552, 64)   16552 timestamps* 30ms /60/1000 = 8.276 minutes
    # print("log mel shape", log_mel.shape)
    # Frame features into examples.
    #features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    #example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))  # 96
    #example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate))  # 7
    #log_mel_examples = mel_features.frame(
    #    log_mel,
    #    window_length=example_window_length,
    #    hop_length=example_hop_length)
    #print(len(data), len(log_mel), len(log_mel_examples))
    #print(len(data) / 16000, params.STFT_WINDOW_LENGTH_SECONDS + len(log_mel) * params.STFT_HOP_LENGTH_SECONDS, params.EXAMPLE_WINDOW_SECONDS + len(log_mel_examples) * params.EXAMPLE_HOP_SECONDS)
    return log_mel


def wavfile_to_examples(
        wav_file, lower_edge_hertz=params.MEL_MIN_HZ, upper_edge_hertz=params.MEL_MAX_HZ):
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype

    data = wav_data / 32768.0    # Convert to [-1.0, +1.0]
    # print("sampling rate:", sr, "wave data", wav_data.shape)
    # (7990639,)

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(data,
                                               audio_sample_rate=sr,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=lower_edge_hertz,
                                               upper_edge_hertz=upper_edge_hertz)
    # (16552, 64)   16552 timestamps* 30ms /60/1000 = 8.276 minutes
    # print("log mel shape", log_mel.shape)
    # Frame features into examples.
    features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))  # 96
    example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate))  # 7
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    #print(len(data), len(log_mel), len(log_mel_examples))
    #print(len(data) / 16000, params.STFT_WINDOW_LENGTH_SECONDS + len(log_mel) * params.STFT_HOP_LENGTH_SECONDS, params.EXAMPLE_WINDOW_SECONDS + len(log_mel_examples) * params.EXAMPLE_HOP_SECONDS)
    return log_mel_examples
