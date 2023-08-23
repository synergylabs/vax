# VAX: Deployment and Testing for commercial sensors

This submodule provides methods to deploy and test multiple commercial sensors independent of running end to end VAX
pipeline. This is a prerequisite for raw data collected for VAX experiments.

## A. Doppler Sensor

We use the [AWR1642BOOST-ODS](https://www.ti.com/tool/AWR1642BOOST-ODS), a 77GHz mmWave evaluation board with an
integrated DSP and an ARM Cortex-R4 Processor from Texas Instruments. The integrated microcontroller on this board is,
however able to provide data at a relatively low frequency of 5Hz, which we overcome by adding a
separate [DCA1000 FPGA board](https://www.ti.com/tool/DCA1000EVM). This board connects directly to the ADCs of the radar
board and extracts the samples of the doppler signals generated in a binary format.

- **Setup and testing AWR1642BOOST**: Find instructions at https://github.com/prasoonpatidar/awr1642boostods_python_sdk
  to install
  mmwave binary (only v3.6.0 supported).
- **Setup testing DCA1000**: The DCA1000 sensor is connected with 60 pin connector with AWR1642 board. We use DCA1000
  CLI
  Interface to setup DCA device, and then use modified [OpenRadar](https://github.com/PreSenseRadar/OpenRadar) API to
  collect and preprocess data in real time. We have provided required binaries and python libraries to setup DCA1000
  board.

## B. Lidar Sensor

We use
the [Slamtec RPLIDAR A1M8](https://www.amazon.com/Slamtec-RPLIDAR-Scanning-Avoidance-Navigation/dp/B07TJW5SXF/ref=asc_df_B07TJW5SXF/)
, a mechanical LiDAR to capture object distance in a single horizontal plane. It is a two-dimensional Lidar sensor with
a maximum range of 6m. The horizontal scanning range is 0°-360°. For more details, here is a link
to [datasheet](https://www.generationrobots.com/media/rplidar-a1m8-360-degree-laser-scanner-development-kit-datasheet-1.pdf)
.

- **Setup and testing:** Connect components for A1M8 as explained in offical guide
  on [A1M8 lidar here](https://www.generationrobots.com/media/rplidar-a1m8-360-degree-laser-scanner-development-kit-datasheet-1.pdf)
  for hardware setup. Find instructions at this repository on software setup and testing for A1M8
  Lidar ([https://github.com/Roboticia/RPLidar/](https://github.com/Roboticia/RPLidar/tree/master))

## C. Thermal Sensor

For thermal sensing, we use
a [FLIR Lepton 3.5](https://www.digikey.com/en/products/detail/flir-lepton/500-0771-01/7606616) thermal
camera,interfaced with
a [Purethermal Mini Development board](https://www.digikey.com/en/products/detail/groupgets-llc/PURETHERMAL-M/9866289)
to enable communication via USB. This sensor captures high-fidelity thermal data (160x120 pixels) at 8Hz. To alleviate
such privacy concerns, we reduced the resolution of the thermal sensor by 16x to just 10x8 pixels, but this can be
changed based on requirements.

- **Setup and testing:** Connect FLIR camera module to dev board, and connect it to your
  system with USB(Make sure to use high speed USB A port (blue ones) and cable). Follow instructions in flirpy
  repository ([https://github.com/LJMUAstroecology/flirpy](https://github.com/LJMUAstroecology/flirpy)) for software
  setup and testing.

## D. Micarray Sensor

For our paper, we used a [ReSpeaker 4-Mic Array](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)
with a Raspberry Pi, which is a quad-microphone expansion board for Raspberry Pi. However, it need an external setup by
connecting our system with an RPI via Ethernet and only allows data flow at 1Hz.

To overcome this, we integrated [Respeaker Micarray v2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) with
our system, which provides usb interface to connect directly with linux based system, thus removing need to integrate
with RPI. Also, it allows for data capture at 100Hz.

- **Micarray V2 Setup:** Follow [instructions here](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) to setup and
  test this device with your system. Make sure to `download 6_channels_firmware.bin` when prompted in instructions.
- **ODAS live Setup:** For realtime Sound Source Localization and Tracking, we
  use [ODAS](https://github.com/introlab/odas). Make sure to build your odaslive binary (at the end of instructions for
  Micarray V2 Setup), and replace it to existing binary `micarray/odaslive_to_replace.bin` [here](micarray/)
  with `$ODAS_SRC_DIR/build/odaslive`.
- **Testing:** Run `cd $CURRENT_DIR/micarray && python3 run_micarray.py` to test whether setup for micarray v2 and
  odaslive. If successful, it should open up a opencv window with realtime visualization for sound localization.

## E. 'DriverInterface' for custom sensor

VAX also provides a driver interface to add custom sensors. You need to copy [driverInteface.py](deviceInterface.py)
to `custom_sensor/run_custom-sensor.py` and implement functions
provided in driverInterface. For implementation details, read through run scripts for existing
sensors for lidar([rplidar](lidar/run_lidar2d.py)), Micarray( [micarray](micarray/run_micarray.py)) and
Thermal ([flir](thermalcam/run_thermal.py)) sensors.

Once you have a working prototype for new sensor, you can write a `custom_sensor/launch_custom_sensor.sh` to include any
system level configuration to run given sensor if needed.

## Need more help??

For more help, feel free to add new github issues, or contact me directly via email [prasoonpatidar@cmu.edu](prasoonpatidar@cmu.edu).

## Reference

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
