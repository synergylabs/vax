# VAX: Using Existing Video and Audio-based Activity Recognition Models to Bootstrap Privacy-Sensitive Sensors

TBD:[[paper (IMWUT 2023)](https://doi.org/10.1145/3610907)]
TBD:[[talk (IMWUT 2023)](https://www.youtube.com/)]
TBD:[[Project Video](https://www.youtube.com/)]
TBD:[[Installation Video](https://www.youtube.com/)]

**Authors:**
[[Prasoon Patidar](http://prasoonpatidar.com/)]
[[Mayank Goel](http://www.mayankgoel.com//)]
[[Yuvraj Agarwal](https://www.synergylabs.org/yuvraj/)]

**Abstract:**
The use of audio and video modalities for Human Activity Recognition (HAR) is common, given the richness of the data and the availability of pre-trained ML models using a large corpus of labeled training data. However, audio and video sensors also lead to significant consumer privacy concerns. Researchers have thus explored alternate modalities that are less privacy-invasive such as mmWave doppler radars, IMUs, motion sensors. However, the key limitation of these approaches is that most of them do not readily generalize across environments and require significant in-situ training data. Recent work has proposed cross-modality transfer learning approaches to alleviate the lack of trained labeled data with some success. In this paper, we generalize this concept to create a novel system called VAX (Video/Audio to ‘X’), where training labels acquired from existing Video/Audio ML models are used to train ML models for a wide range of ‘X’ privacy-sensitive sensors. Notably, in VAX, once the ML models for the privacy-sensitive sensors are trained, with little to no user involvement, the Audio/Video sensors can be removed altogether to protect the user’s privacy better. We built and deployed VAX in ten participants’ homes while they performed 17 common activities of daily living. Our evaluation results show that after training, VAX can use its onboard camera and microphone to detect approximately 15 out of 17 activities with an average accuracy of 90%. For these activities that can be detected using a camera and a microphone, VAX trains a per-home model for the privacy-preserving sensors. These models (average accuracy = 84%) require no in-situ user input. In addition, when VAX is augmented with just one labeled instance for the activities not detected by the VAX A/V pipeline (∼2 out of 17), it can detect all 17 activities with an average accuracy of 84%. Our results show that VAX is significantly better than a baseline supervised-learning approach of using one labeled instance per activity in each home (average accuracy of 79%) since VAX reduces the user burden of providing activity labels by 8x (∼2 labels vs. 17 labels).

**Reference:**
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

## Platform & Hardware requirements:

Our VAX system is deployed and tested on Ubuntu 22.04. Our entire system consists of three different components, and each Artifact of VAX has it's own set of requirements.

- **Data collection pipeline**: To collect data from A/V and privacy-sensitive sensors, we need corresponding sensor hardware along with connectors (either USB or Ethernet Cables). Based on amount of sensors you wish to collect data from, We might need power extension, Ethernet switches etc.
- **Training A/V ensemble with labeled instances**: There is no special hardware setup required. However, it is recommended to have an Nvidia GPU with cuda>=11.3 and cuda<=11.8. 
- **Training Privacy-sensitive sensors**
 
## A. Data collection using VAX Hardware

### Environment Setup:

### 1. Clone (or Fork!) this repository
```
git clone https://github.com/synergylabs/vax.git
```

### 2. Create a virtual environment and install python packages
We recommend using conda. Tested on `Ubuntu 20.04`, with `python 3.7`.

```bash
conda create -n "vax" python=3.9
conda activate vax
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install -r requirements.txt
```
### 3. Sensor Hardware, Setup and testing

#### 3a. Doppler Sensor: 

We use the [AWR1642BOOST-ODS](https://www.ti.com/tool/AWR1642BOOST-ODS), a 77GHz mmWave evaluation board with an integrated DSP and an ARM Cortex-R4 Processor from Texas Instruments. The integrated microcontroller on this board is, however able to provide data at a relatively low frequency of 5Hz, which we overcome by adding a separate [DCA1000 FPGA board](https://www.ti.com/tool/DCA1000EVM). This board connects directly to the ADCs of the radar board and extracts the samples of the doppler signals generated in a binary format.
- **Setup AWR1642BOOST**: Find instructions at https://github.com/prasoonpatidar/awr1642boostods_python_sdk to install mmwave binary (only v3.6.0 supported).
- **Setup DCA1000**: The DCA1000 sensor is connected with 60 pin connector with AWR1642 board. We use DCA1000 CLI Interface to setup DCA device, and then use modified [OpenRadar](https://github.com/PreSenseRadar/OpenRadar) API to collect and preprocess data in real time. We have provided required binaries and python libraries to setup DCA1000 board.
- **Testing:**

#### 3b. Lidar Sensor: 

We use the [Slamtec RPLIDAR A1M8](https://www.amazon.com/Slamtec-RPLIDAR-Scanning-Avoidance-Navigation/dp/B07TJW5SXF/ref=asc_df_B07TJW5SXF/), a mechanical LiDAR to capture object distance in a single horizontal plane. It is a two-dimensional Lidar sensor with a maximum range of 6m. The horizontal scanning range is 0°-360°. For more details, here is a link to [datasheet](https://www.generationrobots.com/media/rplidar-a1m8-360-degree-laser-scanner-development-kit-datasheet-1.pdf).
- **Setup:** 
- **Testing:**

#### 3c. Thermal Sensor:
For thermal sensing, we use a [FLIR Lepton 3.5](https://www.digikey.com/en/products/detail/flir-lepton/500-0771-01/7606616) thermal camera,interfaced with a [Purethermal Mini Development board](https://www.digikey.com/en/products/detail/groupgets-llc/PURETHERMAL-M/9866289) to enable communication via USB. This sensor captures high-fidelity thermal data (160x120 pixels) at 8Hz. To alleviate such privacy concerns, we reduced the resolution of the thermal sensor by 16x to just 10x8 pixels, but this can be changed based on requirements.
- **Setup:** Only setup required for this device is to connect FLIR camera module to dev board, and connect it to your system with USB(Make sure to use high speed USB A port (blue ones) and cable).
- **Testing:**

#### 3d. Micarray Sensor:

For our paper, we used a [ReSpeaker 4-Mic Array](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/) with a Raspberry Pi, which is a quad-microphone expansion board for Raspberry Pi. However, it need an external setup by connecting our system with an RPI via Ethernet and only allows data flow at 1Hz. 

To overcome this, we integrated [Respeaker Micarray v2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) with our system, which provides usb interface to connect directly with linux based system, thus removing need to integrate with RPI. Also, it allows for data capture at 100Hz.

- **Micarray V2 Setup:** Follow [instructions here](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) to setup and test this device with your system. Make sure to download 6_channels_firmware when prompted in instructions.
- **ODAS live Setup:** For realtime Sound Source Localization and Tracking, we use [ODAS](https://github.com/introlab/odas). Make sure to build your odaslive binary (at the end of instructions for Micarray V2 Setup), and replace it to existing binary in ??? location.
- **Testing:**


# Instructions for Running VAX models are coming soon.

### 4. VAX 'DriverInterface' for data collection

### 5. Data Annotation Tool (Streamlit)

### 6. Building Desktop Application (Ubuntu 22.04)

## B. Training A/V ensemble with labelled instances

### 1. Preprocessing raw A/V data

### 2. Running off-the-shelf models on raw data

### 3. Building A/V ensemble using labeled instances

## C. Training and evaluating privacy-sensitive sensors

### 1. Data preprocessing from privacy-sensitive sensors

### 2. Training and evaluating privacy-sensitive sensors

