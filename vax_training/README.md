# VAX Pipeline: Training privacy-sensitive sensors without human supervision for human activity recognition.

## Section A. Workbench Setup

### Environment Setup:

### a. Clone (or Fork!) this repository

```shell
foo@bar:~$ git clone https://github.com/synergylabs/vax.git
```

### b. Create a virtual environment, install python packages and openmm

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.9`.

```shell
foo@bar:~$ conda create --name vax_training python=3.8 -y
foo@bar:~$ conda activate vax_training
foo@bar:~$ pip install -r requirements.txt
```
## Section B. Module Configuration

To train vax models based on A/V labels, you need to provide processed data from X sensors, and final av labels file from ```generate_av_labels``` module output.
You can configure runtime in ```vax_training/vax_training_loocv.py``` in the following lines.

```python
train_config = {
    'av_labels_file': '/home/prasoon/vax/cache/generate_av_labels/temp_user/final_av_labels.csv', # final_av_labels file
    'results_cache_dir': f'/home/prasoon/vax/cache/privacy_models/results',# cache directory to store prediction results
    'training_cache_dir': f'/home/prasoon/vax/cache/privacy_models/training',# cache directory to store trained models
    'base_data_dir':'/home/prasoon/vax/cache/',# base data directory where user's processed data is present
    # No changes required for this part.
    'featurizer': 'm2',
    'trainer': ['svmClean','knnClean'],
    'feature_cache_dir': f'/home/prasoon/vax/cache/privacy_models/features',
    'n_cv_splits': 5, 
    'sensor_list': ['doppler', 'lidar', 'thermal', 'micarray', 'ENV', 'PIR'],
    'sensor_files': ['doppler.pb', 'lidar2d.pb', 'thermal.pb', 'micarray.pb', 'mites.csv', 'mites.csv'],
}
```

## Section C. Running Scripts

After config is setup, to run the script, you have to take following steps from repository root directory:

```shell
foo@bar:~$ cd vax_training/
foo@bar:~$ python vax_training_loocv.py
```

## Section D. Understanding Module Output

The model will output results at an instance level from prediction probabilities using different set of classifiers and sensors, which can be combined using max cumulative score approach. The final results will look as follows: 

```text
instance_id,gt_label,av_label,trainer,prediction_dict,sensor,sensor_label
P1_119,Baking,Blender,adaClean,"{'Blender': 0.5929955296844641, 'Chopping+Grating': 0.4070044703155357}",doppler,Blender
P1_119,Baking,Blender,svmClean,{'Blender': 1.0},doppler,Blender
P1_119,Baking,Blender,knnClean,{'Blender': 1.0},doppler,Blender
P1_121,Baking,Blender,adaClean,{'Blender': 1.0},doppler,Blender
P1_121,Baking,Blender,svmClean,{'Blender': 1.0},doppler,Blender
P1_121,Baking,Blender,knnClean,"{'Blender': 0.9059945504087192, 'CookingOnStove': 0.09400544959128065}",doppler,Blender
P1_120,Baking,Blender,adaClean,{'Blender': 1.0},doppler,Blender
P1_120,Baking,Blender,svmClean,{'Blender': 1.0},doppler,Blender
P1_120,Baking,Blender,knnClean,{'Blender': 1.0},doppler,Blender
P1_122,Baking,Blender,adaClean,{'Blender': 1.0},doppler,Blender
P1_122,Baking,Blender,svmClean,"{'Blender': 0.873379127851175, 'CookingOnStove': 0.12662087214882503}",doppler,Blender
P1_122,Baking,Blender,knnClean,"{'Blender': 0.3810679611650485, 'CookingOnStove': 0.6189320388349514}",doppler,CookingOnStove
P1_116,Baking,Blender,adaClean,{'Chopping+Grating': 1.0},doppler,Chopping+Grating
P1_116,Baking,Blender,svmClean,"{'Blender': 0.8631325316448464, 'Chopping+Grating': 0.025986827449547036, 'WashingDishes': 0.11088064090560665}",doppler,Blender
P1_116,Baking,Blender,knnClean,"{'Blender': 0.7430463576158941, 'Chopping+Grating': 0.25695364238410595}",doppler,Blender
P1_093,Blender,Undetected,adaClean,"{'Blender': 0.20118547554217897, 'CookingOnStove': 0.798814524457821}",doppler,CookingOnStove
P1_093,Blender,Undetected,svmClean,"{'Blender': 0.9203576596865217, 'CookingOnStove': 0.07964234031347826}",doppler,Blender
P1_093,Blender,Undetected,knnClean,"{'Blender': 0.6126005361930295, 'Chopping+Grating': 0.014745308310991957, 'CookingOnStove': 0.3726541554959786}",doppler,Blender
P1_094,Blender,Blender,adaClean,"{'Blender': 0.02283017903875763, 'CookingOnStove': 0.9771698209612425}",doppler,CookingOnStove
P1_094,Blender,Blender,svmClean,"{'Blender': 0.9814539079498866, 'Chopping+Grating': 0.018546092050113264}",doppler,Blender
P1_094,Blender,Blender,knnClean,"{'Blender': 0.9367352252017703, 'Chopping+Grating': 0.005727675084613382, 'CookingOnStove': 0.057537099713616244}",doppler,Blender
P1_095,Blender,Blender,adaClean,"{'Blender': 0.9894575607518753, 'CookingOnStove': 0.010542439248124723}",doppler,Blender
P1_095,Blender,Blender,svmClean,"{'Blender': 0.9666610005845732, 'CookingOnStove': 0.03333899941542675}",doppler,Blender
P1_095,Blender,Blender,knnClean,"{'Blender': 0.8356750152718387, 'CookingOnStove': 0.16432498472816123}",doppler,Blender
P1_097,Blender,Blender,adaClean,"{'Blender': 0.06567621966119465, 'CookingOnStove': 0.9343237803388053}",doppler,CookingOnStove
P1_097,Blender,Blender,svmClean,"{'Blender': 0.9363740679247423, 'Chopping+Grating': 0.04151957221800034, 'CookingOnStove': 0.022106359857257285}",doppler,Blender
P1_097,Blender,Blender,knnClean,"{'Blender': 0.7749320915793558, 'Chopping+Grating': 0.005820721769499418, 'CookingOnStove': 0.21924718665114476}",doppler,Blender```
```

## Reference
For more details, contact [prasoonpatidar@cmu.edu](prasoonpatidar@cmu.edu).

### If you find this module useful in your research, please consider cite:

```bibtex
@INPROCEEDINGS{patidar23vax,
    title = {VAX: Using Existing Video and Audio-based Activity Recognition Models to Bootstrap Privacy-Sensitive Sensors},
    author = {Prasoon Patidar and Mayank Goel and Yuvraj Agarwal},
    journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies}
    year = {2023},
    publisher = {ACM},
    address = {Cancun, Mexico},
    article = {117},
    volume = {7},
    number = {3},
    month = {9},
    doi = {https://doi.org/10.1145/3610907},
    pages = {213–224},
    numpages = {24},
    keywords = { ubiquitous sensing, privacy first design, human activity recognition},
}
```
