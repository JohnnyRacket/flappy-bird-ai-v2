import win32gui
import win32con
import cv2
import time
import webbrowser
from capture_screen import screenshot

print('start debug')

webbrowser.get('C:/Program Files/Google/Chrome/Application/chrome.exe %s').open('https://flappybird.ee/old')
hwnd = win32gui.FindWindow(None, "Play Flappy Bird Online Old - Google Chrome")
win32gui.MoveWindow(hwnd, -8, 0, 700, 616, True)
win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
win32gui.SetForegroundWindow(hwnd)
bbox = win32gui.GetWindowRect(hwnd)
print(bbox)
game_bbox = {"top": bbox[0] + 100, "left": bbox[1] + 186, "width": bbox[2] - 372, "height": bbox[3] - 216}
print(game_bbox) # window now 320 x 400
quit = False

while not quit:
    start_time = time.time()
    img = screenshot(game_bbox)       






    cv2.putText(img, str(int(1.0 / (time.time() - start_time))), (260, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2, cv2.LINE_AA )
    
    cv2.imshow("Debug Classifiers", img)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        quit = True
        print('done debug')