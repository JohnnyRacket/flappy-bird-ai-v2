import cv2
import numpy as np

def resize_and_grayscale_img(img):
    # downsize image to 1/4
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def resize_img(img):
    # downsize image to 1/4
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

def direction_classifier_image_filter(img):
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    #expects 640x360 img
    img = img[40:320, 0:640]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0,140,140])
    upper_red = np.array([2,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([178,140,140])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # add crosshair
    cv2.rectangle(mask, (320-1, 140+1), (320+1, 140-1), (123), 1)
    # result = cv2.bitwise_and(img, img, mask = mask)
    return mask

def gamestate_classifier_filter(img):
    # downsize image to 1/8
    img = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
