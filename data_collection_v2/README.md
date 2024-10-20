# Setup sensing and data collection for VAX-RPI:

## VAX RPI5 Setup:

- Create a new SD card with Raspbian OS
    - Download Raspbian OS from [https://www.raspberrypi.org/software/operating-systems/](https://www.raspberrypi.org/software/operating-systems/)
    - Use Raspberry Pi Imager to flash the SD card.
    
- Setup RPI for user and connect to internet
    - Username: vax
    - Password: vax@123
    - Connect to CMU network using Ethernet cable, and register the device mac address on CMU network.
    - 
        | Hostname: | vax-rpi5-<IDX>.lan.local.cmu.edu |
        | --- | --- |
        | IP address: | xxx.xx.xx.xxx |
        | MAC address: | xx:xx:xx:xx:xx:xx |
        | Network: | xxxx |
- Clone Github repository and copy services
    - Setup git credentials and pull repository in home folder
        - git config —global credentials.helper store
        - git clone https://github.com/prasoonpatidar/vax-rpi
            - Add your credentials
    - Copy linux service to systemctl and enable and start services.
        - ```sudo cp ~/vax-rpi/sensing/services/* /etc/systemd/system```
        - ```sudo daemon-reload```
## Setup TSDB
- Main github repo:[https://github.com/ytyou/ticktock](https://github.com/ytyou/ticktock)
- Main Documentation Guide: [User Guide](https://github.com/ytyou/ticktock/wiki/User-Guide)
    - Install TSDB Image via tar.zip (do not use docker version, it doesn’t work with RPi)
        - [https://github.com/ytyou/ticktock/wiki/User-Guide#1-installation](https://github.com/ytyou/ticktock/wiki/User-Guide#1-installation)
            - See Section **1.3 Install TickTock from a Binary Package**
            - Post installation, See section **1.5 Verify TickTock Running Successfully**
- **Start TSDB service**
    - ```sudo systemctl start ticktockdb.service```
    - Test with ping command. You should get output as a ```pong```.
        - ```curl -XPOST 'http://localhost:6182/api/admin?cmd=ping'```

## Setup for MiniTFT Screen:
- Setup virtual environment for installing custom packages
    - ```python -m venv ~/venv-vax```
    - Add support for system installed packaged in this environment
        - ```nano ~/venv-vax/pyvenv.cfg```
        - set ```include-system-site-packages = true```
    - Install set of basic libraries using apt-get (not supported using pip on RPI)
        - ```sudo apt-get install python3-numpy python3-pandas python3-matplotlib python3-scipy python3-opencv python3-sklearn python3-skimage jq -y```
- Enable RPi interfaces from raspi-config
    - SPI, I2C, SSH.
    - Reboot Pi.
- Install mini-tft screen
    - [https://learn.adafruit.com/adafruit-mini-pitft-135x240-color-tft-add-on-for-raspberry-pi/python-setup](https://learn.adafruit.com/adafruit-mini-pitft-135x240-color-tft-add-on-for-raspberry-pi/python-setup)
        
        
    - Test using basic script if you need to be sure that display is working fine..
    - ISSUE WITH RPI-5:(Although I did a quick fix it in code, Just putting it here for future reference)
        - Github issue: [github.com/adafruit/Adafruit_Blink/issues/75](http://github.com/adafruit/Adafruit_Blink/issues/755)5
            
    Quick Fix: Do not try to access CE0 pin, set cs_pin to None as device is already using this pin(Updated code is checked in already)
            
- **Start stats service**
    - ```sudo systemctl enable record_stats.service```
    - ```sudo systemctl start record_stats.service```
        - You should see stats for Memory/CPU/Disk with time and IP address.
## Setup for Flir
- Clone flirpy repo: [https://github.com/LJMUAstroecology/flirpy](https://github.com/LJMUAstroecology/flirpy)
- Install flirpy in virtual environment
    - ```cd flirpy```
    - ```~/venv-vax/bin/pip install .```
    - ```~/venv-vax/bin/pip install jstyleson```
- Test flirpy with python
    - ```~/venv-vax/bin/python ~/vax-rpi/sensing/record_flir.py```
- If it doesn’t work, adjust the flir module on purethermal mini board. Make sure it is fitted well and not loose.
  - Remove and reconnect usb. You should hear a click and camera module closing and opening again.
      - Try to adjust till you hear the click. Once you do. Try running the script again.
  - if it works, You would see change in flir status on TFT display to red first and then green in ~20 seconds.
  - Stop the script.
- **Start FLIR Service**
    - ```sudo systemctl enable record_flir.service```
    - ```sudo systemctl start record_flir.service```
## Setup for A1M8 Lidar
- Install RPLidar using adafruit.
        - ```venv-vax/bin/pip install adafruit-circuitpython-rplidar sounddevice```
        - ```sudo apt-get install libportaudio2 -y```
- Test run with rplidar script
    - ```~/venv-vax/bin/python ~/vax-rpi/sensing/record_rplidar.py```
    - if it works, You would see change in rplidar status on TFT display to red first and then green in ~20 seconds.
- **Start RPlidar service**
    - ```sudo systemctl enable record_rplidar.service```
    - ```sudo systemctl start record_rplidar.service```
## Setup for Micarray
- Install and build ODAS [https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/#real-time-sound-source-localization-and-tracking](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/#real-time-sound-source-localization-and-tracking)
        - ```sudo apt-get install libfftw3-dev libconfig-dev libasound2-dev libgconf-2-4 libpulse-dev```
        - ```sudo apt-get install cmake```
        - ```cd ~/Desktop```
        - ```git clone https://github.com/introlab/odas.git```
        - ```mkdir odas/build```
        - ```cd odas/build```
        - ```cmake ..```
        - ```make```
- Copy ODAS binary to respeaker folder.
    - cp ~/odas/build/bin/odaslive ~/vax-rpi/sensing/micarray/
    - Test the odaslive script
        - Check which hardware id is respeaker:
            - ~/venv-vax/bin/python -m sounddevice(Look for hw:<deviceID>:0)
            - run ~/vax-rpi/sensing/micarray/respeakerv2/odaslive -c ~/vax-rpi/sensing/micarray/respeakerv2/devices/respeaker_usb_4_mic_array_d<deviceID>.cfg
                - You should see a lot of output printing on terminal continuously.
            - Break the script.
  - Test with vax-rpi script
      - ```~/venv-vax/bin/python ~/vax-rpi/sensing/record_respeakerv2.py```
      - if it works, You would see change in micarray status on TFT display to red first and then green in ~20 seconds.
  - **Start Respeakerv2 service**
      - ```sudo systemctl enable record_respeakerv2.service```
      - ```sudo systemctl start record_respeakerv2.service```
