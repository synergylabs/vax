#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

#allow access to ports
echo "Allowing access to dev ports..."
sudo chmod a+rw /dev/ttyUSB*


# Start Lidar 2D recording
echo "Start Lidar2D Recording"
chmod +x run_lidar2d.py
python run_lidar2d.py &

wait


