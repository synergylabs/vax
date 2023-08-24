# VAX Pipeline: Training privacy-sensitive sensors without human supervision for human activity recognition.

**About:** This tools helps in building an ensemble to map top prediction output from any off-the-shelf A/V model to .

## Section A. Workbench Setup

### Environment Setup:

### a. Clone (or Fork!) this repository

```shell
foo@bar:~$ git clone https://github.com/synergylabs/vax.git
foo@bar:~$ cd vax/data_collection_annotation
```

### b. Create a virtual environment and install python packages

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.9`.

```shell
foo@bar:~$ conda create -n "vax_data_collection" python=3.9
foo@bar:~$ conda activate vax_data_collection
foo@bar:~$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
pytorch-cuda=11.7 -c pytorch -c nvidia
foo@bar:~$ python -m pip install -r requirements.txt
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
