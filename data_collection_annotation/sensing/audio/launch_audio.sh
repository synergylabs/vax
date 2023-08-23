#!/usr/bin/env bash

# Activate conda env
source /home/vax/.bashrc
echo "Activate conda environment"
source /home/vax/anaconda3/bin/activate vax
python --version

#allow access to ports
echo "Allowing access to dev ports..."
sudo chmod a+rw /dev/ttyACM*
sudo chmod a+rw /dev/ttyUSB*

## add dca file to ld library path
echo "Add DCA1000 Binary to library path"
LD_LIBRARY_PATH="/home/vax/vax-codebase/sensing/privacy_sensors/doppler/dca1000"
export LD_LIBRARY_PATH

# Start audio recording
echo "Start Audio Recording"
chmod +x run_audio.py
python run_audio.py &

wait