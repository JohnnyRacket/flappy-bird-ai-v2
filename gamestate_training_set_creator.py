import cv2
import os
import re
import numpy as np
from filter_images import gamestate_classifier_filter

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def load_images_from_folder(folder):
    images = []
    for filename in sorted_alphanumeric(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

imgs = load_images_from_folder(r'screen_recording')

# os.chdir(r'C:\Users\John\Videos')
for index, img in enumerate(imgs):
    img = gamestate_classifier_filter(img)
    cv2.imwrite((r'gamestate_classifier\gamestate_classifier_imgs\img' + str(index + 1) + '.png'), img)
