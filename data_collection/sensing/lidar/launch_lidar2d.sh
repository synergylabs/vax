#!/usr/bin/env bash

# add data collection annotation directory to sensing
echo "Add data_collection_annotation to root directory"
export PYTHONPATH="${PYTHONPATH}:${HOME}/vax/data_collection_annotation/"
CONFIG_FILE="$HOME/vax/data_collection_annotation/config/data_collection_config.json"

# Activate conda env
source $HOME/.bashrc
echo "Activate conda environment"
source $HOME/anaconda3/bin/activate vax_data_collection
python --version

#allow access to ports
echo "Allowing access to dev ports..."
sudo chmod a+rw /dev/ttyUSB*

# Start Lidar 2D recording
echo "Start Lidar2D Recording"
chmod +x sensing/lidar/run_lidar2d.py
python sensing/lidar/run_lidar2d.py $CONFIG_FILE &

wait


