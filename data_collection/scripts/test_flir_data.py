import _pickle
import binascii

import pandas as pd
import numpy as np
import base64
import pickle
import seaborn
import glob
from datetime import datetime, timedelta
import time
from dateutil import parser
import cv2
from dateutil import tz
import os
import sys
import json
import matplotlib.pyplot as plt

thermal_file = "/Users/ppatida2/VAX/DataCollection/phase3/processed_data/P11/UsingOven/P11_051/thermal.pb"
thermal_data = pickle.load(open(thermal_file, "rb"))
thermal_frame_1 = thermal_data[0][1]

# remove any negative values and convert it into nan
thermal_frame_1[thermal_frame_1 < 0] = np.nan

# fill any nan values with average of all neighbouring pixels
thermal_frame_1 = cv2.inpaint(thermal_frame_1, np.isnan(thermal_frame_1).astype(np.uint8), 3, cv2.INPAINT_TELEA)


print("finished")