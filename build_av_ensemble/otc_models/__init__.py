"""
This package consists code for running off the shelf models
"""

from HAR.Phase2IntegratedNN.otc_models.posec3d import *
from HAR.Phase2IntegratedNN.otc_models.yamnet import *

model_init_funcs = {
    'posec3d_ntu120': posec3d_ntu120_model,
    'posec3d_ntu60': posec3d_ntu60_model,
    'posec3d_hmdb': posec3d_hmdb_model,
    'posec3d_ucf': posec3d_ucf_model,
    'stgcn_ntu60': stgcn_ntu60_model,
    'yamnet':get_yamnet_model
}


def get_model(model_id, device='cpu'):
    model, class_names = model_init_funcs[model_id](device)
    return model, class_names

