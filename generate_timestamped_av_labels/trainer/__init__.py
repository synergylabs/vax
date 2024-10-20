"""
This consist of wide variety of trainers for training models for privacy preserving sensors
"""
from .xgbBasic import privacy_fit as xgbBasic, get_clf_ as xgbBasicClf
from .adaClean import privacy_fit as adaClean, get_clf_ as adaClf
from .svmClean import privacy_fit as svmClean, get_clf_ as svmClf
from .knnClean import privacy_fit as knnClean, get_clf_ as knnClf
from .voteClean import privacy_fit as voteClean, get_clf_ as voteClf
from .xgbClean import privacy_fit as xgbClean, get_clf_ as xgbClf

# from .xgbIsolation import crossvalidatePVSSensors as xgbIsolationCV
# from .xgbSmote import crossvalidatePVSSensors as xgbSmoteCV

pvs_trainer_map = {
    'xgbBasic': xgbBasic,
    'adaClean': adaClean,
    'svmClean': svmClean,
    'voteClean': voteClean,
    'knnClean': knnClean,
    'xgbClean': xgbClean,
    # 'xgbIsolation': xgbIsolationCV,
    # 'xgbSmote': xgbSmoteCV

}

pvs_clf_map = {
    'adaClean': adaClf,
    'svmClean': svmClf,
    'voteClean': voteClf,
    'knnClean': knnClf,
    'xgbClean': xgbClf,
    # 'xgbIsolation': xgbIsolationCV,
    # 'xgbSmote': xgbSmoteCV

}


def get_trainer(trainer_name):
    return pvs_trainer_map[trainer_name]

def get_clf(trainer_name):
    return pvs_clf_map[trainer_name]