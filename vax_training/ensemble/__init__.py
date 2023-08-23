"""
This consist of wide variety of ensemble methods for combining output from sensor models
"""
from HAR.privacy_sensors.ml.ensemble.score_ensemble import getEnsembleOutput as score_ensemble
from HAR.privacy_sensors.ml.ensemble.norm_ensemble import getEnsembleOutput as norm_ensemble

pvs_ensemble_map = {
    'score_ensemble': score_ensemble,
    'norm_ensemble': norm_ensemble
}


def get_ensemble(ensemble_name):
    return pvs_ensemble_map[ensemble_name]
