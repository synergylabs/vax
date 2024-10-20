
import time

from vl53l5cx.vl53l5cx import VL53L5CX
from vl53l5cx.api import VL53L5CX_RESOLUTION_8X8, VL53L5CX_RANGING_MODE_AUTONOMOUS,VL53L5CX_POWER_MODE_WAKEUP, VL53L5CX_TARGET_ORDER_CLOSEST
import numpy as np
driver = VL53L5CX(nb_target_per_zone=1, disable_ambient_per_spad=True, disable_nb_spads_enabled=True,disable_signal_per_spad=True)


alive = driver.is_alive()
if not alive:
    raise IOError("VL53L5CX Device is not alive")

print("Initialising...")
t = time.time()
driver.init()
driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
driver.set_power_mode(VL53L5CX_POWER_MODE_WAKEUP)
driver.set_ranging_frequency_hz(15)
driver.set_ranging_mode(VL53L5CX_POWER_MODE_WAKEUP)
driver.set_target_order(VL53L5CX_TARGET_ORDER_CLOSEST)

print(f"Initialised ({time.time() - t:.1f}s)")


# Ranging:
driver.start_ranging()
print("Ranging Started")
previous_time = 0
loop = 0
frame_data = []
num_frames = 0
st_time = time.time()

while True:
    if driver.check_data_ready():
        ranging_data = driver.get_ranging_data()
        num_frames +=1
        if ranging_data.nb_of_detected_aggregates > 0:
            print("\n\n\n\n\n")
            print(ranging_data.motion)
            print("\n\n\n\n\n")
        if (time.time()-st_time) > 10:
            print(num_frames, len(ranging_data.distance_mm), len(ranging_data.range_sigma_mm), len(ranging_data.reflectance), ranging_data.motion[::-1])
            # ~ print(",".join(list(map(str,map(int, ranging_data.distance_mm)))))
            # ~ print(",".join(list(map(str,map(int, ranging_data.range_sigma_mm)))))
            # ~ print(",".join(list(map(str,ranging_data.reflectance))))
            motion_arr = np.flipud(np.array(list(ranging_data.motion_indicator.motion))).astype('float64')
            num_frames=0.
            st_time = time.time()
