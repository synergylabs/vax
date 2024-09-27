
import time

from vl53l5cx.vl53l5cx import VL53L5CX
from vl53l5cx.api import VL53L5CX_RESOLUTION_8X8

driver = VL53L5CX()


alive = driver.is_alive()
if not alive:
    raise IOError("VL53L5CX Device is not alive")

print("Initialising...")
t = time.time()
driver.init()
driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
driver.set_
print(f"Initialised ({time.time() - t:.1f}s)")


# Ranging:
driver.start_ranging()
previous_time = 0
loop = 0
while loop < 100:
    if driver.check_data_ready():
        ranging_data = driver.get_ranging_data()

        # As the sensor is set in 4x4 mode by default, we have a total 
        # of 16 zones to print. For this example, only the data of first zone are 
        # print
        now = time.time()
        if previous_time != 0:
            time_to_get_new_data = now - previous_time
            print(f"Print data no : {driver.streamcount: >3d} ({time_to_get_new_data * 1000:.1f}ms)")
        else:
            print(f"Print data no : {driver.streamcount: >3d}")
		
        for i in range(8):
            zone_row = ""
            for j in range(8):
                zone_id = 8*i + j
                zone_distance = ranging_data.distance_mm[driver.nb_target_per_zone * zone_id]
                if zone_distance < 50:
                    zone_sym = '.'
                elif zone_distance < 250:
                    zone_sym = '*'
                else:
                    zone_sym = '+'
                zone_row += ' ' + zone_sym
            print(zone_row)
        print("")

        previous_time = now
        loop += 1

    #time.sleep(3)
