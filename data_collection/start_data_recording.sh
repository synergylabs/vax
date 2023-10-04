#!/usr/bin/env bash

# launch interactive dbus
if test -z "$DBUS_SESSION_BUS_ADDRESS" ; then
  exec dbus-run-session -- bash
  echo "D-Bus per-session daemon address is: $DBUS_SESSION_BUS_ADDRESS"
fi

# add data collection annotation directory to sensing
echo "Add data_collection_annotation to root directory"
export PYTHONPATH="${PYTHONPATH}:${HOME}/vax/data_collection/"
CONFIG_FILE="$HOME/vax/data_collection/config/data_collection_config.json"

# Activate conda env
source $HOME/.bashrc
echo "Activate conda environment"
source $HOME/anaconda3/bin/activate vax_data_collection
python --version

#allow access to ports
echo "Allowing access to dev ports..."
sudo chmod a+rw /dev/ttyACM*
sudo chmod a+rw /dev/ttyUSB*

## add dca file to ld library path
echo "Add DCA1000 Binary to library path"
LD_LIBRARY_PATH="$HOME/vax/data_collection/sensing/doppler/dca1000"
export LD_LIBRARY_PATH
export PATH="$PATH:$HOME/vax/data_collection/sensing/doppler/dca1000"

# Start Thermal recording
echo "Start Thermal Recording"
chmod +x sensing/thermalcam/run_thermal.py
python sensing/thermalcam/run_thermal.py $CONFIG_FILE &

sleep 5


# Start Lidar 2D recording
echo "Start Lidar2D Recording"
chmod +x sensing/lidar/run_lidar2d.py
python sensing/lidar/run_lidar2d.py $CONFIG_FILE &

sleep 5

# Start Micarray recording
echo "Start Micarray Recording"
chmod +x sensing/micarray/run_micarray.py
python sensing/micarray/run_micarray.py $CONFIG_FILE &

sleep 5


# Start doppler recording
echo "Start Doppler Recording"
chmod +x sensing/doppler/run_doppler_awr1642.py
# Starting a gnome terminal manually: /usr/bin/dbus-launch /usr/bin/gnome-terminal &
python sensing/doppler/run_doppler_awr1642.py $CONFIG_FILE &
sleep 5

wait


