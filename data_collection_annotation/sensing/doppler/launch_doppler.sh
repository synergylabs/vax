#!/usr/bin/env bash

# launch interactive dbus
if test -z "$DBUS_SESSION_BUS_ADDRESS" ; then
  exec dbus-run-session -- bash
  echo "D-Bus per-session daemon address is: $DBUS_SESSION_BUS_ADDRESS"
fi

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

# Start doppler recording
echo "Start Doppler Recording"
chmod +x run_doppler.py
# Starting a gnome terminal manually: /usr/bin/dbus-launch /usr/bin/gnome-terminal &
python run_doppler.py &
# /usr/bin/dbus-launch /usr/bin/gnome-terminal -x bash -c "python run_doppler.py;exec bash" &

wait


