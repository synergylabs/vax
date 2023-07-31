import numpy as np
import cv2
from flirpy.camera.lepton import Lepton
import time

# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__=='__main__':
    with Lepton() as camera:
        while True:
            img = camera.grab()
            # Rescale to 8 bit
            if img is not None:
                img = img.astype(np.float32)
                img = 255 * (img - img.min()) / (img.max() - img.min())
                # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
                # You can also try PLASMA or MAGMA
                img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

                # resizing image for better visualization
                scale_percent = 300  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img_resized = cv2.resize(img_col, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow('Lepton', img_resized)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    #
    # cv2.destroyAllWindows()
