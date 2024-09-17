'''
Config class for strict parameters for training optics models
'''
import os
ROOT_DIR = '..'
if not os.path.exists(ROOT_DIR):
    ROOT_DIR = '..'
class OpticsConfig:
    raw_models = {
        #     'samosa':f'{ROOT_DIR/vax/cache/av_results_23_1_23/samosa/*',
        'yamnet': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/yamnet/*',
        'posec3d_hmdb': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/posec3d/hmdb/*',
        'posec3d_ntu60': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/posec3d/ntu60/*',
        'posec3d_ntu120': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/posec3d/ntu120/*',
        'posec3d_ucf': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/posec3d/ucf/*',
        'stgcn_ntu60': f'{ROOT_DIR}/vax/cache/av_results_23_1_23/stgcn/ntu60/*',
    }

    label_filters = {
        #     'samosa':['Other'],
        'yamnet': ['Silence', 'Inside, small room', 'Speech', 'Inside, large room or hall'],
        'posec3d_ntu120': ['staggering'],
        'posec3d_ntu60': ['staggering'],
        'posec3d_ucf': [],
        'posec3d_hmdb': [],
        'stgcn_ntu60': [],
    }
    activity_filters = ['Sitting', 'Shower', 'WatchingTV', 'Toothbrush','Talking','Knocking']
    instance_filters = ['P1_069', 'P1_079', 'P1_076', 'P1_114',
                        'P2_028',
                        'P3_119',
                        'P4_051', 'P4_058', 'P4_118', 'P4_125',
                        'P5_018', 'P5_028', 'P5_075', 'P5_000', 'P5_006', 'P5_015', 'P5_024', 'P5_030',
                        'P6_007', 'P6_037', 'P6_053']
    audio_models = ['samosa', 'yamnet']

    samosa_context_labels_map = {
        'Kitchen': [
            'Blender_in_use',
            'Chopping',
            'Grating',
            'Microwave',
            'Twisting_jar',
            'Washing_Utensils',
            'Washing_hands',
        ],
        'Bathroom': [
            'Brushing_hair',
            'Hair_dryer_in_use',
            'Shaver_in_use',
            'Toilet_flushing',
            'Toothbrushing',
            'Washing_hands',
            'Wiping_with_rag',
        ],
        'LivingRoom': [
            'Clapping',
            'Coughing',
            'Drinking',
            'Knocking',
            'Laughing',
            'Pouring_pitcher',
            'Shaver_in_use',
            'Twisting_jar',
            'Vacuum in use',
        ]
    }
    activity_context_map = {
        'Baking': 'Kitchen',
        'Blender': 'Kitchen',
        'Chopping+Grating': 'Kitchen',
        #     'Chopping': 'Kitchen',
        'CookingOnStove': 'Kitchen',
        'FridgeOpen': 'Kitchen',
        #     'Grating': 'Kitchen',
        'Microwave': 'Kitchen',
        'WashingDishes': 'Kitchen',

        'Coughing': 'LivingRoom',
        # 'Drinking': 'LivingRoom',
        # 'Eating': 'LivingRoom',
        'Drinking/Eating':'LivingRoom',
        'Exercising': 'LivingRoom',
        'Knocking': 'LivingRoom',
        # 'Sitting': 'LivingRoom',
        'Talking': 'LivingRoom',
        'Vacuum': 'LivingRoom',
        'Walking': 'LivingRoom',
        #     'WatchingTV': 'LivingRoom',

        'HairBrush': 'Bathroom',
        'HairDryer': 'Bathroom',
        'HandWash': 'Bathroom',
        'Shaver In Use': 'Bathroom',
        #     'Shower': 'Bathroom',
        'ToilerFlushing': 'Bathroom',
        #     'Toothbrush': 'Bathroom',

    }
    context_activities = {
        'Kitchen': ['Baking', 'Blender', 'Chopping+Grating', 'Chopping', 'CookingOnStove', 'FridgeOpen', 'Grating',
                    'Microwave', 'WashingDishes'],
        'Bathroom': ['HairBrush', 'HairDryer', 'HandWash', 'Shaver In Use', 'Shower', 'ToilerFlushing', 'Toothbrush'],
        'LivingRoom': ['Coughing', 'Drinking', 'Eating', 'Drinking/Eating','Exercising', 'Knocking', 'Talking', 'Vacuum', 'Walking',
                       'WatchingTV']
    }
