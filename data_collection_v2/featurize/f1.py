"""
Live Featurization for multiple sensors
"""

import numpy as np
import pandas as pd
import skimage.measure


def get_features(raw_input, sensor, featurizer='m2'):

    sensor_featurize = None
    if sensor=='rplidar':
        sensor_featurize = featurize_rplidar
    elif sensor=='flir':
        sensor_featurize = featurize_flir
    elif sensor=='respeakerv2':
        sensor_featurize = featurize_respeakerv2
    elif sensor=='mlx':
        sensor_featurize = featurize_mlx
    elif sensor=='tofimager':
        sensor_featurize = featurize_tofimager
    else:
        raise NotImplementedError("Sensor not available...")

    ts_sensor, X_sensor = sensor_featurize(raw_input)
    return ts_sensor, X_sensor


def featurize_rplidar(instance_data):
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, lidar_data in instance_data:
        lidar_data_x, lidar_data_y = lidar_data
        lidar_data_x, lidar_data_y = np.array(lidar_data_x), np.array(lidar_data_y)
        lidar_data_r = np.sqrt((lidar_data_x ** 2) + (lidar_data_y ** 2))
        lidar_data_th = np.arctan2(lidar_data_y, lidar_data_x) * 180 / np.pi
        sort_indices = np.argsort(lidar_data_th)
        lidar_data_th = lidar_data_th[sort_indices]
        lidar_data_r = lidar_data_r[sort_indices]
        bin_count, _ = np.histogram(lidar_data_th, range(-180, 180))
        bin_idx = np.digitize(lidar_data_th, range(-180, 180)) - 180
        df_inst = pd.DataFrame(np.array([bin_idx, lidar_data_r]).T, columns=['angle', 'distance'])
        # remove less than 200 cms to remove corners and rig interference
        # df_inst = df_inst[df_inst.distance>200]
        df_inst = df_inst.groupby('angle', as_index=False).agg({'distance': 'mean'})
        df_inst = df_inst.astype(int)
        # df_inst['distance'] = (df_inst['distance']//100)*100
        df_inst = pd.merge(pd.DataFrame(range(-180, 180), columns=['angle']), df_inst[['angle', 'distance']],
                           how='left').fillna(0.)
        instant_feature_vec.append(df_inst.distance.values)

    instant_feature_vec = np.array(instant_feature_vec)
    instant_avg_vec = np.mean(instant_feature_vec, axis=0)
    boundary_mask = np.absolute(instant_avg_vec - instant_feature_vec) <= 100
    instant_feature_vec[boundary_mask] = 0.
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


def featurize_flir(instance_data, thermal_compression_=16):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and thermal data
    :return:
    """
    instance_data_compressed = []
    for i in range(len(instance_data)):
        instance_data_compressed.append(
            (instance_data[i][0], skimage.measure.block_reduce(instance_data[i][1], (
                thermal_compression_, thermal_compression_), np.max)))

    timestamps = np.array([xr[0] for xr in instance_data_compressed])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, thermal_data in instance_data_compressed:
        instant_feature_vec.append(thermal_data.flatten())

    instant_feature_vec = np.array(instant_feature_vec)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        median_instance_vals = np.median(instant_feature_vec[timestamp_secs == ts], axis=0)
        min_instance_vals = np.min(instant_feature_vec[timestamp_secs == ts], axis=0)
        sec_feature_vec.append(np.concatenate([median_instance_vals, min_instance_vals]))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec


def cyl2sph(x, y, z):
    # z = rcos(phi)
    # x = rsin(phi)cos(theta)
    # y = rsin(phi)sin(theta)
    if x == y == z == 0:
        return 0, np.nan, np.nan
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    theta_sign = np.sign(np.arcsin(y / (r * np.sin(phi))))
    theta_ = np.arccos(x / (r * np.sin(phi)))
    return r, phi * 180 / np.pi, theta_sign * theta_ * 180 / np.pi


def featurize_ss(ss_dict):
    res = np.zeros(49)
    res[-1] = ss_dict.get('activity', ss_dict.get('E', 0))
    r, phi, theta = cyl2sph(ss_dict['x'], ss_dict['y'], ss_dict['z'])
    if r > 0:
        # localize in 45 degree cones
        flat_cone_idx = np.digitize([theta], bins=range(-180, 180, 45))
        z_cone_idx = np.digitize([phi], bins=range(0, 180, 45))
        cone_idx = (4 * flat_cone_idx) + z_cone_idx
        res[cone_idx] = r
    return res


def featurize_respeakerv2(instance_data):
    """
    Returns featurization based on max in sample and median across sample at 1 sec level
    :param instance_data: list of tuples with ts and micarray data
    :return:
    """
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, micarray_data in instance_data:
        # micarray_data_sst, micarray_data_ssl = micarray_data['SST'].values(), micarray_data['SSL'].values()
        feature_ssts = np.array([featurize_ss(xr) for xr in micarray_data['SST']])
        # Method 1: ssts only
        instant_feature_vec.append(np.sum(feature_ssts, axis=0))
        # # Method 2: combine ssls and ssts
        # feature_ssls = np.array([featurize_ss(xr) for xr in micarray_data['SSL']])
        # instant_feature_vec.append(np.sum(feature_ssts + (0.25 * feature_ssls), axis=0))
    instant_feature_vec = np.array(instant_feature_vec)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec

def featurize_tofimager(instance_data):
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, data in instance_data:
        motion_arr = np.flipud(np.array(list(data.motion_indicator.motion))).astype('float64').flatten()
        distance_arr = np.flipud(np.array(data.distance_mm)).astype('float64').flatten()
        reflectance_arr = np.flipud(np.array(data.reflectance)).astype('float64').flatten()
        feature_vals = np.concatenate([distance_arr, reflectance_arr, motion_arr])
        instant_feature_vec.append(feature_vals)
    instant_feature_vec = np.array(instant_feature_vec)
    instant_avg_vec = np.mean(instant_feature_vec, axis=0)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec    

def featurize_tofimager_old(instance_data):
    timestamps = np.array([xr[0] for xr in instance_data])
    timestamp_secs = timestamps // 1e9
    instant_feature_vec = []
    for instance_ts, ranging_data in instance_data:
        ranging_distance, ranging_sigma, ranging_reflectance = np.array(ranging_data.distance_mm), np.array(ranging_data.range_sigma_mm), np.array(ranging_data.reflectance)
        feature_vals = np.concatenate([ranging_distance, ranging_sigma, ranging_reflectance])
        instant_feature_vec.append(feature_vals)
    instant_feature_vec = np.array(instant_feature_vec)
    instant_avg_vec = np.mean(instant_feature_vec, axis=0)
    sec_feature_vec = []
    sec_ts = []
    for ts in np.unique(timestamp_secs):
        sec_ts.append(ts)
        sec_feature_vec.append(np.mean(instant_feature_vec[timestamp_secs == ts], axis=0))
    sec_feature_vec = np.array(sec_feature_vec)
    return np.array(sec_ts), sec_feature_vec

