import cv2
import multiprocessing
from capture_screen import screenshot
import win32gui
import win32con
import webbrowser


def save(queue):
    number = 0
    while 'there are screenshots':
        img = queue.get()
        if img is None:
            print('done recording')
            break
        cv2.imwrite((r'screen_recording\img' + str(number + 1) + '.png'), img)
        number += 1


if __name__ == '__main__':

    print('start screen record')

    # webbrowser.get('C:/Program Files/Google/Chrome/Application/chrome.exe %s').open('https://flappybird.ee/old')
    hwnd = win32gui.FindWindow(None, "Play Flappy Bird Online Old - Google Chrome")
    win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
    win32gui.MoveWindow(hwnd, -8, 0, 980, 616, True)
    win32gui.SetForegroundWindow(hwnd)

    quit = False
    queue = multiprocessing.Queue()
    multiprocessing.Process(target=save, args=(queue,)).start()

    while not quit:
        bbox = {'top': 147, 'left': 196, 'width': 320, 'height': 420}
        img = screenshot(bbox)       
        queue.put(img)

        cv2.imshow("Record Screen", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            quit = True
            queue.put(None)
            print('finish screen record')
            

