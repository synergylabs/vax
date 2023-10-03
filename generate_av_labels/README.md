# VAX: A/V ensemble to combine outputs from any set of off-the-shelf models for human activity recognition.

**About:** This tools helps in building an ensemble to map top prediction output from any off-the-shelf A/V model to . 

## Section A. Workbench Setup

### Environment Setup:

### a. Clone (or Fork!) this repository

```shell
foo@bar:~$ git clone https://github.com/synergylabs/vax.git
foo@bar:~$ cd vax/data_collection
```

### b. Create a virtual environment, install python packages and openmm

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.9`.

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
**Details coming soon...**

## Section B. Module Configuration

**Details coming soon...**

## Section C. Running Scripts

**Details coming soon...**

## Section D. Understanding Module Output

**Details coming soon...**



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
