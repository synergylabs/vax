from run import readAndParseData1642Boost, stopDoppler, startDoppler
import traceback
import numpy as np
import cv2
import tempfile
import time
awr1642_filename = tempfile.mktemp(prefix='delme_awr1642_unlimited_',
                                     suffix='.csv', dir='')
file_handler = open(awr1642_filename,'w')
cliport, dataport, configdict = startDoppler()
while True:
    try:
        dataOk, frameNumber, detObj = readAndParseData1642Boost(dataport, configdict)
        if dataOk:
            img =detObj['rangeDopplerMatrix']
            file_handler.write(f'{str(time.time_ns())} | {str(img)}')
            img = img.astype(np.float32)
            img = 255 * (img - img.min()) / (img.max() - img.min())
            img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

            # resizing image for better visualization
            scale_percent = 800  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Doppler', img_resized)
        if cv2.waitKey(1) == 27:
            print("Closing Doppler")
            stopDoppler(cliport, dataport)
            break  # esc to quit
    except:
        print("Closing Doppler")
        print(traceback.print_exc())
        stopDoppler(cliport,dataport)
