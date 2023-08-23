# VAX: Data Preprocessing and Data Visualization

**About:**??

## Section A. Workbench Setup

### A.1: Environment Setup:

### 1. Clone (or Fork!) this repository

```
git clone https://github.com/synergylabs/vax.git
cd vax/data_preprocessing_visualization
```

### 2. Create a virtual environment and install python packages

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.9`.

```bash
conda create -n "vax_data_preprocessing" python=3.9
conda activate vax_data_preprocessing
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install -r requirements.txt
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
