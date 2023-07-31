#!/usr/bin/env bash

if [[ $USER == "vax" ]]; then
    sudo $0
fi

cd /home/vax/vax-codebase


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

# Start Terminating processes file
echo "Start Process termination"
chmod +x kill_sensors.py
/usr/bin/gnome-terminal -x bash -c "python kill_sensors.py;exec bash" &

wait


