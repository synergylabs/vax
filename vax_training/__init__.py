
activity_context_map = {
    'Baking': 'Kitchen',
    'Blender': 'Kitchen',
    'Chopping+Grating': 'Kitchen',
    # 'Chopping': 'Kitchen',
    'CookingOnStove': 'Kitchen',
    'FridgeOpen': 'Kitchen',
    # 'Grating': 'Kitchen',
    'Microwave': 'Kitchen',
    'WashingDishes': 'Kitchen',

    'Coughing': 'LivingRoom',
    'Drinking/Eating':'LivingRoom',
    'Exercising': 'LivingRoom',
    'Knocking': 'LivingRoom',
    # 'Talking': 'LivingRoom',
    'Vacuum': 'LivingRoom',
    'Walking': 'LivingRoom',
    # 'WatchingTV': 'LivingRoom',

    'HairBrush': 'Bathroom',
    'HairDryer': 'Bathroom',
    'HandWash': 'Bathroom',
    'Shaver In Use': 'Bathroom',
    # 'Shower': 'Bathroom',
    'ToilerFlushing': 'Bathroom',
    # 'Toothbrush': 'Bathroom',

}

context_activities={
    'Kitchen':['Baking','Blender','Chopping+Grating','CookingOnStove','FridgeOpen','Grating','Microwave','WashingDishes'],
    'Bathroom':['HairBrush','HairDryer','HandWash','Shaver In Use','ToilerFlushing'],
    'LivingRoom':['Coughing','Drinking/Eating','Exercising','Knocking','Vacuum','Walking']
}

sensor_list= ['doppler', 'lidar', 'thermal', 'micarray','FeatureMIC','IMU', 'EMI', 'ENV', 'PIR', 'GridEye', 'WIFI']
activities = ['Undetected'] + list(activity_context_map.keys())

instance_filters = ['P1_069','P1_079','P1_076','P1_114',
                         'P2_028',
                         'P3_119',
                         'P4_051','P4_058','P4_118','P4_125',
                         'P5_018','P5_028','P5_075','P5_000','P5_006','P5_015','P5_024','P5_030',
                         'P6_007','P6_037','P6_053']
