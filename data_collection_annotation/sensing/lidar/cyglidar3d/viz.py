from run import readAndParseDataCygbot, start3DLidar, stop3DLidar
import traceback
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import tempfile

serialObj = start3DLidar()
st_time = time.time()
max_duration_in_secs = 60
lidar3d_filename = tempfile.mktemp(prefix='delme_lidar3d_unlimited_',
                                     suffix='.csv', dir='')
f_handler = open(lidar3d_filename,'w')
while True:
    try:
        dataOk, detObj = readAndParseDataCygbot(serialObj)
        if dataOk:
            img =detObj['mat_3d']
            f_handler.write(f'{str(time.time_ns())} | {str(img)}')
            img = img.astype(np.float32)
            img = 255 * (img - img.min()) / (img.max() - img.min())
            img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_BONE)

            # resizing image for better visualization
            scale_percent = 300  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('3D Lidar', img_resized)
            if time.time() > st_time+max_duration_in_secs:
                stop3DLidar(serialObj)
                time.sleep(2)
                serialObj = start3DLidar()
                st_time = time.time()
        if cv2.waitKey(1) == 27:
            print("Closing 3D Lidar")
            stop3DLidar(serialObj)
            break  # esc to quit
    except:
        print("Closing 3D Lidar")
        print(traceback.print_exc())
        stop3DLidar(serialObj)
