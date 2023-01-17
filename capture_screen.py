
import mss
import numpy as np

def screenshot(bbox):
    with mss.mss() as sct:

        img = np.array(sct.grab(bbox))
        return img