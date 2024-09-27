
import time
import numpy as np
import cv2
from vl53l5cx.vl53l5cx import VL53L5CX
from vl53l5cx.api import VL53L5CX_RESOLUTION_8X8, VL53L5CX_RANGING_MODE_AUTONOMOUS,VL53L5CX_POWER_MODE_WAKEUP, VL53L5CX_TARGET_ORDER_CLOSEST

driver = VL53L5CX(nb_target_per_zone=1, disable_ambient_per_spad=True, disable_nb_spads_enabled=True,disable_signal_per_spad=True)


alive = driver.is_alive()
if not alive:
    raise IOError("VL53L5CX Device is not alive")

print("Initialising...")
t = time.time()
driver = VL53L5CX(nb_target_per_zone=1, disable_ambient_per_spad=True, disable_nb_spads_enabled=True,disable_signal_per_spad=True)
driver.init()
driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
driver.set_power_mode(VL53L5CX_POWER_MODE_WAKEUP)
driver.set_ranging_frequency_hz(15)
driver.set_ranging_mode(VL53L5CX_POWER_MODE_WAKEUP)
driver.set_target_order(VL53L5CX_TARGET_ORDER_CLOSEST)
driver.set_sharpener_percent(90)


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
		img = np.array(ranging_data.distance_mm, dtype=np.float32).reshape(8,8)
		img = 255* (img-img.min()) / (img.max() - img.min())
		img_col = cv2.applyColorMap(img.astype(np.uint8),cv2.COLORMAP_INFERNO)
		img_col = cv2.flip(img_col, 1)
		# resizing image for better visualization
		scale_percent = 15000  # percent of original size
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_LINEAR)
		cv2.imshow('TOF', img_resized)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
		num_frames +=1

		if (time.time()-st_time) > 10:
			print(num_frames, len(ranging_data.distance_mm), len(ranging_data.range_sigma_mm), len(ranging_data.reflectance), ranging_data.nb_of_detected_aggregates)
			num_frames=0.
			st_time = time.time()
			
