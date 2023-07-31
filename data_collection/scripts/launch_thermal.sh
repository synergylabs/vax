#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

# Start Thermal recording
echo "Start Thermal Recording"
chmod +x run_thermal.py
python run_thermal.py &

sleep 2

wait


