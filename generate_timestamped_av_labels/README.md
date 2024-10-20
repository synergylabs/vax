# VAX: Get A/V labels by combining outputs from a set of off-the-shelf models for human activity recognition.

## Section A. Workbench Setup

### Environment Setup:

### a. Clone (or Fork!) this repository

```shell
foo@bar:~$ git clone https://github.com/synergylabs/vax.git
```

### b. Create a virtual environment, install python packages and openmm

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.8`.

```shell
foo@bar:~$ conda create --name vax_av_labels python=3.8 -y
foo@bar:~$ conda activate vax_av_labels
foo@bar:~$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
foo@bar:~$ pip install -r requirements.txt
foo@bar:~$ pip install mmcv==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html
foo@bar:~$ pip install mmpose==0.29.0
foo@bar:~$ pip install -U openmim
foo@bar:~$ mim install mmengine
foo@bar:~$ mim install mmdet==2.28.2
foo@bar:~$ pip install -e git+https://github.com/open-mmlab/mmaction2.git@0c6182f8007ae78b512d9dd7320ca76cb1cfd938#egg=mmaction2
```
## Section B. Module Configuration

To generate AV labels, you need to provide processed data from X sensors, and for each instance in processed data, you need to add an ```camera.mp4``` file
which will be video recording for the captured instance.

Once you have .mp4 file across all instances, You can configure runtime in ```generate_av_labels/generate_av_labels.py``` in the following lines.

```python
# config for pipeline run
BASE_SRC_DIR = '/home/prasoon/vax/' # Base directory for vax repo
BASE_DATA_DIR = 'cache/temp_user/processed_data' # Base directory for processed data
USER = 'temp_user' # username

run_config = {
    'cache_dir': f'{BASE_SRC_DIR}/cache/generate_av_labels/{USER}', #cache directory 
    'activity_set': ['Baking', 'Blender', 'Chopping+Grating', 'CookingOnStove', 'FridgeOpen',
                     'Microwave', 'WashingDishes'],# list of activities you wish to create model on
    'user' : USER,
    'run_av_only':True,
    'run_id': f'{USER}_test_e2e',
    # No changes needed after this point
    'run_context': 'Kitchen',
    'data_dir':f'{BASE_DATA_DIR}',
    'av_ensemble_file': f'{BASE_SRC_DIR}/generate_av_labels/av_ensemble.pb',
    'featurizer': 'm2',
}
```

## Section C. Running Scripts

After config is setup, to run the script, you have to take following steps from repository root directory:

```shell
foo@bar:~$ cd generate_av_labels/
foo@bar:~$ python generate_av_labels.py
```

## Section D. Understanding Module Output

The final output of model will be a csv file (```final_av_labels.csv```) with generated AV labels in cache directory set in Section B.
which looks as follows, and consists of detailed labeling output across A/V ensemble and "X" clustering.

```text
instance_id,audio_prediction,audio_score,video_prediction,video_score,doppler_prediction,doppler_score,lidar_prediction,lidar_score,thermal_prediction,thermal_score,condensed_prediction,condensed_score,final_prediction,final_score
P3_007,WashingDishes,0.0023885570233120206,Blender,0.1427166873850139,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.645913321678976,WashingDishes,0.8634040164266066,Undetected,0.0
P3_008,Blender,3.2186847588433284e-05,Microwave,0.1884869406325868,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.645913321678976,Blender,0.8184560003804459,Undetected,0.0
P3_009,Baking,0.1517843282863976,Baking,0.1520727755722904,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.951844999643844,Undetected,0.0
P3_010,Baking,0.03555837684777816,Microwave,0.22972452620520648,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.645913321678976,Baking,0.7809393968522953,Undetected,0.0
```
This is the input file needed for training vax models in next step.

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
