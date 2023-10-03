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

#al
# Start Micarray recording
echo "Start Micarray Recording"
chmod +x sensing/micarray/run_micarray.py
python sensing/micarray/run_micarray.py $CONFIG_FILE &

wait


