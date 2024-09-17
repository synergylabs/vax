'''

'''
import pandas as pd
import numpy as np
import os
import pickle
# mmwave for noise reduction
# import mmwave.dsp as dsp
# import mmwave.clustering as clu

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.cluster import *

from datetime import datetime

from utils import time_diff
from sensor_clusterer import *

sensor_clusterer = {
    'doppler': cluster_doppler_data,
    'thermal': cluster_thermal_data,
    'lidar': cluster_lidar_data,
    'micarray': cluster_micarray_data,
    'PIR': cluster_PIR_data,
    'ENV': cluster_ENV_data,
}

def cluster_training_data(vax_pipeline_object, config, logger):
    cluster_training_data_ts_start = datetime.now()

    training_cluster_cache = f'{config["cache_dir"]}/x_cluster_info.pb'
    if os.path.exists(training_cluster_cache):
        sensor_predictions = pickle.load(open(training_cluster_cache,"rb"))
    else:
        raw_sensor_data = vax_pipeline_object['raw_sensor_data']
        raw_video_output = vax_pipeline_object['raw_av_labels']['video']
        raw_audio_output = vax_pipeline_object['raw_av_labels']['audio']
        condensed_prediction = vax_pipeline_object['raw_av_labels']['condensed']

        threshold_info = config['thresholds']

        # loop over contexts and sensors to create instance level prediction for each sensor
        sensor_predictions = {sensor: dict() for sensor in config['sensor_list']}
        # sensor_predictions = {}

        for sensor in sensor_predictions.keys():
            if sensor not in raw_sensor_data.keys():
                continue
            logger.info(f"Getting {sensor} support...")
            id_train, ts_train, X_train = raw_sensor_data[sensor]

            sensor_cluster_labels = sensor_clusterer[sensor](X_train)
            _, counts = np.unique(sensor_cluster_labels, return_counts=True)
            # membership_fraction = 1 - (counts[0] / sum(counts))
            # sensor_cluster_count = max(sensor_cluster_labels) + 1
            df_cluster = pd.DataFrame(zip(id_train, sensor_cluster_labels), columns=['instance_id', 'cluster_id'])
            df_cluster['value'] = 1.
            df_cluster = pd.merge(df_cluster.groupby(['instance_id', 'cluster_id'], as_index=False)['value'].sum(),
                                  df_cluster.groupby(['instance_id'], as_index=False)['value'].sum(),
                                  on=['instance_id'], suffixes=('', '_total'))
            df_cluster['id_fraction'] = df_cluster['value'] / df_cluster['value_total']
            df_cluster = df_cluster[(df_cluster.cluster_id >= 0) & (
                    df_cluster.id_fraction >= threshold_info['x_min_instance_cluster_overlap'])]
            df_cluster = pd.merge(df_cluster,
                                  df_cluster.groupby(['cluster_id'], as_index=False)['instance_id'].count(),
                                  on=['cluster_id'], suffixes=('', '_total'))
            # df_cluster = df_cluster[df_cluster.instance_id_total <= threshold_info['x_max_cluster_instance_count']]
            for cluster_id in df_cluster.cluster_id.unique():
                cluster_instances = df_cluster[df_cluster.cluster_id == cluster_id].instance_id.unique().tolist()
                cluster_predictions = merge_dicts(
                    [(condensed_prediction[inst] if inst in condensed_prediction.keys() else {}) for inst in
                     cluster_instances])
                cluster_predictions = {xr: cluster_predictions[xr] / sum(cluster_predictions.values()) for xr in
                                       cluster_predictions}
                sensor_predictions[sensor].update(
                    {clu_instance: cluster_predictions for clu_instance in cluster_instances})
        pickle.dump(sensor_predictions, open(training_cluster_cache,"wb"))
    cluster_training_data_ts_end = datetime.now()

    logger.info(
        f"Got training clusters in {time_diff(cluster_training_data_ts_start, cluster_training_data_ts_end)}")

    return sensor_predictions