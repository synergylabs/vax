#!/usr/bin/env bash


# Start Terminating processes file
echo "Start Process termination"
chmod +x kill_sensors.py
/usr/bin/gnome-terminal -x bash -c "python kill_sensors.py;exec bash" &

wait


