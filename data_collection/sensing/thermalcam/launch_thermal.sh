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

# Start Thermal recording
echo "Start Thermal Recording"
chmod +x sensing/thermalcam/run_thermal.py
python sensing/thermalcam/run_thermal.py $CONFIG_FILE &

sleep 2

wait


