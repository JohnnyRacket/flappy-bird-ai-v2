import win32api
import win32con
import time

def press_spacebar():
    win32api.keybd_event(0x20, 0,0,0)
    # time.sleep(.05)
    win32api.keybd_event(0x20,0 ,win32con.KEYEVENTF_KEYUP ,0)