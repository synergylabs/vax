"""
Different modules to featurize data from privacy preserving sensors
"""
import HAR.privacy_sensors.ml.featurize.m1 as featurize_m1
import HAR.privacy_sensors.ml.featurize.m2 as featurize_m2

featurize_module_map = {
    'm1': featurize_m1,
    'm2': featurize_m2
}


def get_features(raw_input, sensor, featurizer='m2'):
    """
    The get_features function takes in a raw input and sensor type, and returns the time series of the features
    and the feature matrix. The featurizer is determined by featurize_module_map[featurizer]. For example, if
    featurizer='m2', then we will use m2's doppler module to get features from a Doppler sensor.

    Args:
        raw_input: Pass the raw data to the featurizer
        sensor: Specify the sensor type
        featurizer='m2': Select the featurizer module

    Returns:
        The time series and feature matrix for a given sensor

    Doc Author:
        Trelent
    """

    featurize_module =  featurize_module_map[featurizer]

    sensor_featurize = None
    if sensor=='doppler':
        sensor_featurize = featurize_module.doppler
    elif sensor=='lidar':
        sensor_featurize = featurize_module.lidar
    elif sensor=='thermal':
        sensor_featurize = featurize_module.thermal
    elif sensor=='micarray':
        sensor_featurize = featurize_module.micarray
    elif sensor=='IMU':
        sensor_featurize = featurize_module.IMU
    elif sensor=='EMI':
        sensor_featurize = featurize_module.EMI
    elif sensor=='ENV':
        sensor_featurize = featurize_module.ENV
    elif sensor=='PIR':
        sensor_featurize = featurize_module.PIR
    elif sensor=='GridEye':
        sensor_featurize = featurize_module.GridEye
    elif sensor=='WIFI':
        sensor_featurize = featurize_module.WIFI
    elif sensor=='FeatureMIC':
        sensor_featurize = featurize_module.FeatureMIC
    else:
        print("Sensor not available...")

    ts_sensor, X_sensor = sensor_featurize(raw_input)
    return ts_sensor, X_sensor