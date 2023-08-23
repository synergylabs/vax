#!/usr/bin/env bash

# Changing Directory
#sleep 30
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

# Start Status file
echo "Start Data Collection Stats"
chmod +x data_collection_stats.py
/usr/bin/gnome-terminal -x bash -c "python data_collection_stats.py;exec bash" &
echo "Sleeping for 2 secs"
sleep 2

# Start audio recording
echo "Start Audio Recording"
chmod +x run_audio.py
python run_audio.py &

echo "Sleeping for 5 secs"
sleep 5

# Start video recording
echo "Start Video Recording"
chmod +x run_oakdlite.py
python run_oakdlite.py &
sleep 5

# Start video recording
#echo "Start Video Recording"
#chmod +x run_video.py
#python run_video.py &
#sleep 5


# Start Thermal recording
echo "Start Thermal Recording"
chmod +x run_thermal.py
python run_thermal.py &

sleep 2


# Start Lidar 2D recording
echo "Start Lidar2D Recording"
chmod +x run_lidar2d.py
python run_lidar2d.py &

sleep 5
#
#Start Lidar 3D recording
#echo "Start Lidar3D Recording"
#chmod +x run_lidar3d.py
#python run_lidar3d.py &
#
#sleep 5

# Start Micarray recording
echo "Start Micarray Recording"
chmod +x run_micarrayv2.py
python run_micarrayv2.py &

sleep 5


# Start doppler recording
echo "Start Doppler Recording"
chmod +x run_doppler.py
#python run_doppler.py &
/usr/bin/dbus-launch /usr/bin/gnome-terminal -x bash -c "python run_doppler.py;exec bash" &

sleep 5

wait


