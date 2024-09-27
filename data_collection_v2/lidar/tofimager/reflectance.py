#!/usr/bin/env python3

import time
import vl53l5cx_ctypes as vl53l5cx
import numpy
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm


COLOR_MAP = "twilight"
INVERSE = True


def get_palette(name):
    cmap = cm.get_cmap(name, 256)

    try:
        colors = cmap.colors
    except AttributeError:
        colors = numpy.array([cmap(i) for i in range(256)], dtype=float)

    arr = numpy.array(colors * 255).astype('uint8')
    arr = arr.reshape((16, 16, 4))
    arr = arr[:, :, 0:3]
    return arr.tobytes()


pal = get_palette(COLOR_MAP)

print("Uploading firmware, please wait...")
vl53 = vl53l5cx.VL53L5CX()
print("Done!")
vl53.set_resolution(8 * 8)

# This is a visual demo, so prefer speed over accuracy
vl53.set_ranging_frequency_hz(15)
vl53.set_integration_time_ms(5)
vl53.start_ranging()


while True:
    if vl53.data_ready():
        data = vl53.get_data()
        arr = numpy.flipud(numpy.array(data.reflectance).reshape((8, 8))).astype('float64')

        # Scale reflectance (a percentage) to 0 - 255
        arr *= (255.0 / 100.0)
        arr = numpy.clip(arr, 0, 255)

        # Invert the array : 0 - 255 becomes 255 - 0
        if INVERSE:
            arr *= -1
            arr += 255.0

        # Force to int
        arr = arr.astype('uint8')

        # Convert to a palette type image
        img = Image.frombytes("P", (8, 8), arr)
        img.putpalette(pal)
        img = img.convert("RGB")
        img_resized = np.array(img.resize((240, 240), resample=Image.NEAREST))

        # Display the result
        cv2.imshow('TOF', img_resized)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    time.sleep(0.01)  # Avoid polling *too* fast
