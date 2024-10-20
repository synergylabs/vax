'''
Inference engine for yamnet model
'''
from __future__ import division, print_function


import numpy as np
import resampy
import otc_models.model_configs.yamnet.params as yamnet_params
import otc_models.model_configs.yamnet.yamnet as yamnet_model
import pandas as pd
from pathlib import Path

def get_yamnet_model(device='cpu'):
    model_file = 'model_ckpts/yamnet.h5'
    class_names_file = 'model_configs/yamnet/yamnet_class_map.csv'
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(f"{Path(__file__).parent}/{model_file}")
    yamnet_classes = yamnet_model.class_names(f"{Path(__file__).parent}/{class_names_file}")
    return yamnet, yamnet_classes

def audio_inference(instance_audio_data, yamnet, yamnet_classes):
    params = yamnet_params.Params()
    # wav_data, sr = sf.read(audio_file, dtype=np.int16)
    wav_data, sr = instance_audio_data
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    df_scores = pd.DataFrame(scores.numpy().T, index=yamnet_classes)
    return df_scores
