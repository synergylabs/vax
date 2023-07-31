from run import startMicarray, stopMicarray
import traceback
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tempfile
import time
micarray_filename = tempfile.mktemp(prefix='delme_micarray_unlimited_',
                                     suffix='.csv', dir='')
file_handler = open(micarray_filename,'w')
micarray_proc = startMicarray()
ID_IDX = 0
X_IDX, Y_IDX, Z_IDX = 1, 2, 3
ACT_IDX = 4
colors = [(255, 0, 0), (0, 255, 0),
          (0, 0, 255), (255, 255, 255)]

try:
    while True:
        detObj = micarray_proc.read()
        file_handler.write(f'{str(time.time_ns())} | {str(detObj)}')
        sst_dict = detObj['SST']
        sources = []
        for mic_src in sst_dict.keys():
            if not (sst_dict[mic_src][ID_IDX] == 0):
                sources.append(mic_src)
        # print(detObj['Timestamp'])

        # Get SSL sources for energy calculation
        ssl_dict = detObj['SSL']
        ssl_sources = []
        for mic_src in ssl_dict.keys():
            if not (ssl_dict[mic_src][-1] == 0.):
                ssl_sources.append(mic_src)

        window_dimension = 400
        multipler = window_dimension // 4
        cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
        cv2_img = cv2.line(cv2_img, (0, window_dimension // 2), (window_dimension, window_dimension // 2),
                           (255, 255, 255), 4)
        cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0), (window_dimension // 2, window_dimension),
                           (255, 255, 255), 4)
        for mic_src in sources:
            idx, x, y, z, _ = sst_dict[mic_src]
            px = min((window_dimension // 2) - int(x * multipler), window_dimension)
            py = min((window_dimension // 2) + int(y * multipler), window_dimension)
            z_size = int(np.abs(z) * 10)
            z_sign = np.sign(z)
            if z_sign < 0:
                cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
            else:
                cv2_img = cv2.circle(cv2_img, (px, py), z_size, (255, 255, 255), -1)
            # print(px,py,z_size)

        # Draw Circles for SSL Sources
        for mic_src in ssl_sources:
            x, y, z, e = ssl_dict[mic_src]
            px = min((window_dimension // 2) - int(x * multipler), window_dimension)
            py = min((window_dimension // 2) + int(y * multipler), window_dimension)
            e = min(e, 1)
            # point_color = (int(255 * e), 0, int(255 * (1 - e)))
            point_color = (255, 255, 255)
            z_size = int(np.abs(z) * 10)
            z_sign = np.sign(z)
            if z_sign < 0:
                cv2_img = cv2.circle(cv2_img, (px, py), z_size, point_color, z_size // 4)
            else:
                cv2_img = cv2.circle(cv2_img, (px, py), z_size, point_color, z_size // 4)

        img_col = cv2.applyColorMap(cv2_img.astype(np.uint8), cv2.COLORMAP_OCEAN)
        # print(img_col.shape)
        scale_percent = 100  # percent of original size
        width = int(cv2_img.shape[1] * scale_percent / 100)
        height = int(cv2_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
        # print(img_resized.shape)
        cv2.imshow('Micarray', img_resized)
        if cv2.waitKey(1) == 27:
            print("Closing Micarray")
            stopMicarray(micarray_proc)
            # stop2DLidar(serialObj)
            break  # esc to quit



except KeyboardInterrupt:
    stopMicarray(micarray_proc)

    # window_dimension = 400
    # multipler = 8000 // window_dimension
    # cv2_img = np.zeros((window_dimension, window_dimension), dtype=np.float32)
    # cv2_img = cv2.line(cv2_img, (0, window_dimension // 2), (window_dimension, window_dimension // 2),
    #                    (255, 255, 255), 4)
    # cv2_img = cv2.line(cv2_img, (window_dimension // 2, 0), (window_dimension // 2, window_dimension),
    #                    (255, 255, 255), 4)

    # try:
    #     img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_BONE)
    #     # resizing image for better visualization
    #     scale_percent = 400  # percent of original size
    #     width = int(img.shape[1] * scale_percent / 100)
    #     height = int(img.shape[0] * scale_percent / 100)
    #     dim = (width, height)
    #     img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
    #     cv2.imshow('3D Lidar', img_resized)
    #     if cv2.waitKey(1) == 27:
    #         print("Closing 3D Lidar")
    #         stop3DLidar(serialObj)
    #         break  # esc to quit
    # except:
    #     print("Closing 3D Lidar")
    #     print(traceback.print_exc())
    #     stop3DLidar(serialObj)
