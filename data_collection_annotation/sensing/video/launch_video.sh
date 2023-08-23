#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

#allow access to ports
echo "Allowing access to dev ports..."
chmod a+rw /dev/ttyACM*
chmod a+rw /dev/ttyUSB*

# Start video recording
echo "Start Video Recording"
chmod +x run_video.py
python run_video.py &
sleep 5


wait


