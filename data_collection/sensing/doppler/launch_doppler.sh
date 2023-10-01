#!/usr/bin/env bash

# launch interactive dbus
if test -z "$DBUS_SESSION_BUS_ADDRESS" ; then
  exec dbus-run-session -- bash
  echo "D-Bus per-session daemon address is: $DBUS_SESSION_BUS_ADDRESS"
fi

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
sudo chmod a+rw /dev/ttyACM*
sudo chmod a+rw /dev/ttyUSB*

## add dca file to ld library path
echo "Add DCA1000 Binary to library path"
LD_LIBRARY_PATH="$HOME/vax/data_collection_annotation/sensing/doppler/dca1000"
export LD_LIBRARY_PATH
export PATH="$PATH:$HOME/vax/data_collection_annotation/sensing/doppler/dca1000"

# Start doppler recording
echo "Start Doppler Recording"
chmod +x sensing/doppler/run_doppler.py
# Starting a gnome terminal manually: /usr/bin/dbus-launch /usr/bin/gnome-terminal &
python sensing/doppler/run_doppler.py $CONFIG_FILE &
# /usr/bin/dbus-launch /usr/bin/gnome-terminal -x bash -c "python run_doppler.py;exec bash" &

wait


