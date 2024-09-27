#!/usr/bin/env bash

# add data collection annotation directory to sensing
echo "Add data_collection to root directory"
export PYTHONPATH="${PYTHONPATH}:${HOME}/vax/data_collection/"
CONFIG_FILE="$HOME/vax/data_collection/config/data_collection_config.json"

#allow access to ports
echo "Allowing access to dev ports..."
sudo chmod a+rw /dev/ttyUSB*

# Start Lidar 2D recording
echo "Start Lidar2D Recording"
chmod +x record_rplidar.py
python record_rplidar.py &

wait


