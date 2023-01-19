import win32gui
import win32con
import cv2
import time
import webbrowser
import numpy as np
from capture_screen import screenshot

from gamestate_classifier import gamestate_classifier
from crash_classifier import crash_classifier
from filter_images import gamestate_classifier_filter, crash_classifier_filter, penalty_classifier_filter, rl_agent_filter

print('start debug')

# webbrowser.get('C:/Program Files/Google/Chrome/Application/chrome.exe %s').open('https://flappybird.ee/old')
hwnd = win32gui.FindWindow(None, "Play Flappy Bird Online Old - Google Chrome")
win32gui.MoveWindow(hwnd, -8, 0, 980, 616, True)
win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
win32gui.SetForegroundWindow(hwnd)
bbox = win32gui.GetWindowRect(hwnd)
print(bbox)
game_bbox = {"top": bbox[0] + 155, "left": bbox[1] + 196, "width": bbox[2] - 652, "height": bbox[3] - 196}
print(game_bbox) # window now 320 x 400
quit = False

x = np.linspace(-1, 1, 80)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)

while not quit:
    start_time = time.time()
    img = screenshot(game_bbox)

    img2 = gamestate_classifier_filter(img[20:,:])
    gamestate = gamestate_classifier.classify_gamestate(img2)

    img3 = penalty_classifier_filter(img)
    # conv_coord = np.dstack((img3, xv, yv)).astype(np.uint8)
    is_crashing =  crash_classifier.classify_gamestate(img3)
    img = rl_agent_filter(img)
    # if(is_crashing == 'crashing'):
    #     print('crashing')
    
    # if(gamestate == 'title_screen'):
    #     print('--------------------------')

    cv2.putText(img, str(gamestate), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (200,0,255), 2, cv2.LINE_AA )

    cv2.putText(img, str(is_crashing), (160, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (200,100,100), 2, cv2.LINE_AA )
    time.sleep(max(1./33 - (time.time() - start_time), 0))
    cv2.putText(img, str(int(1.0 / (time.time() - start_time))), (260, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2, cv2.LINE_AA )
    img = cv2.resize(img, (0, 0), fx=4, fy=4)
    # cv2.putText(img3, str(is_crashing), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (200,100,100), 2, cv2.LINE_AA )
    cv2.imshow("Debug Classifiers", img)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        quit = True
        print('done debug')
    