#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

#allow access to ports
echo "Allowing access to dev ports..."
#chmod a+rw /dev/ttyACM*
chmod a+rw /dev/ttyUSB*

#Start Lidar 3D recording
echo "Start Lidar3D Recording"
chmod +x run_lidar3d.py
python run_lidar3d.py &

sleep 5
wait