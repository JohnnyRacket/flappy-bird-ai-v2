import cv2
import numpy as np

def resize_and_grayscale_img(img):
    # downsize image to 1/4
    new_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # convert to grayscale
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    return new_img

def resize_img(img):
    # downsize image to 1/4
    new_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGRA2BGR)

    return new_img

def rl_agent_filter(img):
    new_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # lower mask (0-10)
    # lower_red = np.array([0,140,140])
    # upper_red = np.array([2,255,255])
    # mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # # upper mask (170-180)
    # lower_red = np.array([178,140,140])
    # upper_red = np.array([180,255,255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # # join my masks
    # mask = mask0+mask1

    return new_img

def gamestate_classifier_filter(img):
    # downsize image to 1/8
    new_img = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
    # convert to grayscale
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    return new_img
