'''
This is config dict for basic configuration for running vax pipeline, which do not change frequently
'''

base_config = {
    # fallback configuration
    'run_id': 'test_e2e',
    'run_context': 'Kitchen',
    'cache_dir': f'/Users/ppatida2/VAX/vax/cache/phase2_e2e',
    'activity_set': ['Baking', 'Blender', 'Chopping+Grating', 'CookingOnStove', 'FridgeOpen',
                     'Microwave', 'WashingDishes'],
    'featurizer': 'm2',
    # AV Labeling configurations
    'instance_window_length': 10,  # in seconds
    'instance_sliding_length': 5,  # in seconds
    'pose_otc_model_names': ['posec3d_ntu120',
                             'posec3d_ntu60',
                             'posec3d_hmdb',
                             'posec3d_ucf',
                             'stgcn_ntu60'],
    'audio_otc_model_names': ['yamnet'],

    'label_filters' : {
    #     'samosa':['Other'],
    'yamnet': ['Silence', 'Inside, small room', 'Inside, large room or hall'],
    'posec3d_ntu120': ['staggering'],
    'posec3d_ntu60': ['staggering'],
    'posec3d_ucf': [],
    'posec3d_hmdb': [],
    'stgcn_ntu60': [],
    },
    'activity_filters' : ['Sitting', 'Shower', 'WatchingTV', 'Toothbrush', 'Talking', 'Knocking'],
    'thresholds': {
        # av only thresholds
        'audio_high': 0.9,
        'audio_low': 0.4,
        'video_high': 2.5,
        'video_low': 0.5,

        # conf classes threshold
        # 'cf_gmm_components': 3,
        # 'cf_max_confusion_classes': 2,
        # 'cf_min_optics_membership_fraction': 0.5,
        # 'cf_min_optics_clusters': 2,
        # 'cf_min_optics_samples': 30,
        # 'cf_max_confusion_clusters': 3,
        # 'cf_min_instance_cluster_overlap': 0.3,

        # X support thresholds
        'x_min_instance_cluster_overlap': 0.5,
        # 'x_max_cluster_instance_count': 5,
        'x_raw_av_min_score': 0.95,
        'x_thermal_min_score': 0.8,
        'x_lidar_min_score': 0.8,
        'x_micarray_min_score': 0.8,
        'x_doppler_min_score': 0.5,
        'x_PIR_min_score': 0.5,
        'x_ENV_min_score': 0.75,

    },
    'activity_context_map': {
        'Baking': 'Kitchen',
        'Blender': 'Kitchen',
        'Chopping+Grating': 'Kitchen',
        'CookingOnStove': 'Kitchen',
        'FridgeOpen': 'Kitchen',
        'Microwave': 'Kitchen',
        'WashingDishes': 'Kitchen',

        'Coughing': 'LivingRoom',
        'Drinking/Eating': 'LivingRoom',
        'Exercising': 'LivingRoom',
        'Knocking': 'LivingRoom',
        'Vacuum': 'LivingRoom',
        'Walking': 'LivingRoom',

        'HairBrush': 'Bathroom',
        'HairDryer': 'Bathroom',
        'HandWash': 'Bathroom',
        'Shaver In Use': 'Bathroom',
        'ToilerFlushing': 'Bathroom',

    },
    'activities': ['Undetected', 'Baking', 'Blender', 'Chopping+Grating', 'CookingOnStove', 'FridgeOpen', 'Grating',
                   'Microwave',
                   'WashingDishes', 'HairBrush', 'HairDryer', 'HandWash', 'Shaver In Use', 'ToilerFlushing',
                   'Coughing', 'Drinking/Eating', 'Exercising', 'Knocking', 'Vacuum', 'Walking'],
    'context_activities_map': {
        'Kitchen': ['Baking', 'Blender', 'Chopping+Grating', 'CookingOnStove', 'FridgeOpen', 'Grating', 'Microwave',
                    'WashingDishes'],
        'Bathroom': ['HairBrush', 'HairDryer', 'HandWash', 'Shaver In Use', 'ToilerFlushing'],
        'LivingRoom': ['Coughing', 'Drinking/Eating', 'Exercising', 'Knocking', 'Vacuum', 'Walking']
    },
    'video_models': ('yamnet', 'posec3d_hmdb', 'posec3d_ntu60', 'posec3d_ntu120', 'posec3d_ucf',
                   'stgcn_ntu60'),
    'audio_models': ('yamnet',),

    # X sensor training contexts
    'trainer': ['adaClean', 'knnClean', 'svmClean'],
    'n_cv_splits': 5,
    'sensor_list': ['doppler', 'lidar', 'thermal'],
    'sensor_files': ['doppler.pb', 'lidar2d.pb', 'thermal.pb', 'micarray.pb', 'mites.csv', 'mites.csv'],
}
