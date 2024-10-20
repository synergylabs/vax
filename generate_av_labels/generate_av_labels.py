"""
This is main package file, which starts vax pipeline based on given configuration
"""
from datetime import datetime
import os.path
import pandas as pd
from copy import deepcopy
import sys

from utils import get_logger
from base_config import base_config
from get_instance_information import get_instance_raw_av_data
from get_otc_output import get_otc_output
from get_raw_av_labels import get_raw_av_labels
from featurize_training_data import featurize_training_data
from cluster_training_data import cluster_training_data
from combine_av_clustering import combine_av_clustering

def run_vax_av_pipeline(config, logger):
    logger.info(f"----- End to End VAX pipeline run for user {config['user']} -----")

    # Create instances from continuous timestamp data
    logger.info("Create instances from continuous timestamp data")
    vax_pipeline_object = get_instance_raw_av_data(config, logger)

    # Get OTC model output over raw A/V data
    logger.info("Get OTC output from raw audio and pose data")
    otc_labels = get_otc_output(vax_pipeline_object, config, logger)
    vax_pipeline_object['otc_labels'] = otc_labels

    # Get A/V labels from training ensemble
    logger.info("Get raw av labels from otc model data")
    raw_av_labels = get_raw_av_labels(vax_pipeline_object, config, logger)
    vax_pipeline_object['raw_av_labels'] = raw_av_labels


    # Featurize training data from X sensors for given instances
    logger.info("Featurize data from X sensors for given instances")
    featurized_training_data, raw_sensor_data = featurize_training_data(vax_pipeline_object, config, logger)
    vax_pipeline_object['featurized_training_data'] = featurized_training_data
    vax_pipeline_object['raw_sensor_data'] = raw_sensor_data

    # Cluster featurized training data from X sensors
    logger.info("Cluster featurized training data from X sensors")
    sensor_predictions = cluster_training_data(vax_pipeline_object, config, logger)
    vax_pipeline_object['sensor_predictions'] = sensor_predictions

    # Update A/V labels from clustering input
    logger.info("Combine clustering information with raw_av_labels to get final av labels")
    final_av_labels = combine_av_clustering(vax_pipeline_object, config, logger)
    vax_pipeline_object['final_av_labels'] = final_av_labels
    df_av_labels  = pd.DataFrame.from_dict(final_av_labels, orient='index').reset_index()
    df_av_labels = df_av_labels.rename(columns={'index':'instance_id'})
    final_av_labels_file = f'{config["cache_dir"]}/final_av_labels.csv'
    df_av_labels.to_csv(final_av_labels_file,index=False)
    return None


if __name__ == '__main__':
    # config for pipeline run
    BASE_SRC_DIR = '../'
    USER = 'P11'
    BASE_DATA_DIR = f'../../../mnt/vax/phase3/{USER}'

    run_config = {
        'user' : USER,
        'run_av_only':True,
        'run_id': f'{USER}_test_e2e',
        'num_training_days':2,
        'run_context': 'Kitchen',
        'data_dir':f'{BASE_DATA_DIR}',
        'av_ensemble_file': f'{BASE_SRC_DIR}/generate_av_labels/av_ensemble.pb',
        'cache_dir': f'{BASE_SRC_DIR}/generate_av_labels/cache/phase2_e2e/{USER}',
        'activity_set': ['Baking', 'Blender', 'Chopping+Grating', 'CookingOnStove', 'FridgeOpen',
                         'Microwave', 'WashingDishes'],
        'featurizer': 'm2',
    }

    config = deepcopy(base_config)
    config.update(run_config)

    # Get logger
    curr_time = datetime.now().strftime("%Y%m%d_%H")
    logger = get_logger(f"{config['run_id']}_{curr_time}")

    # check if base directory, data directory and ensemble file exists
    if not os.path.exists(BASE_SRC_DIR):
        logger.info(f"Base directory {BASE_SRC_DIR} does not exist. Exiting...")
        sys.exit(1)

    if not os.path.exists(run_config['data_dir']):
        logger.info(f"Raw data directory {config['data_dir']} does not exist. Exiting...")
        sys.exit(1)

    if not os.path.exists(run_config['av_ensemble_file']):
        logger.info("A/V ensemble does not exist. Exiting...")
        sys.exit(1)

    # Create cache directory
    if not os.path.exists(run_config['cache_dir']):
        os.makedirs(run_config['cache_dir'])

    # run vax pipeline
    run_vax_av_pipeline(config, logger)

