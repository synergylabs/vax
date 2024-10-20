#!/usr/bin/env python3

import time
import vl53l5cx_ctypes as vl53l5cx
import numpy as np
COLOR_MAP = "plasma"
INVERSE = False


print("Uploading firmware, please wait...")
vl53 = vl53l5cx.VL53L5CX()
print("Done!")
vl53.set_resolution(8 * 8)
vl53.set_power_mode('Wakeup')
# Enable motion indication at 8x8 resolution
vl53.enable_motion_indicator(8 * 8)

# Default motion distance is quite far, set a sensible range
# eg: 40cm to 1.4m
vl53.set_motion_distance(1000, 2000)

# This is a visual demo, so prefer speed over accuracy
vl53.set_ranging_frequency_hz(15)
vl53.set_integration_time_ms(5)
vl53.start_ranging()

print("Ranging Started")
previous_time = 0
loop = 0
frame_data = []
num_frames = 0
st_time = time.time()
while True:
    if vl53.data_ready():
        data = vl53.get_data()
        num_frames +=1
        if (time.time()-st_time) > 10:
            print(num_frames)
            motion_arr = np.flipud(np.array(list(data.motion_indicator.motion))).astype('float64').flatten()
            distance_arr = np.flipud(np.array(data.distance_mm)).astype('float64').flatten()
            reflectance_arr = np.flipud(np.array(data.reflectance)).astype('float64').flatten()
            # ~ print(motion_arr)
            # ~ print(distance_arr)
            # ~ print(reflectance_arr)
            print(np.concatenate([distance_arr, reflectance_arr, motion_arr]))
            num_frames=0.
            st_time = time.time()
    time.sleep(0.01)  # Avoid polling *too* fast
