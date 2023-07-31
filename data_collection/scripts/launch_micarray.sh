#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

# Start Micarray recording
echo "Start Micarray Recording"
chmod +x run_micarrayv2.py
python run_micarrayv2.py &

wait


