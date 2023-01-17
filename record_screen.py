import cv2
import multiprocessing
from capture_screen import screenshot

def save(queue):
    number = 0
    while "there are screenshots":
        img = queue.get()
        if img is None:
            print('done recording')
            break
        cv2.imwrite((r'screen_recording\img' + str(number + 1) + '.png'), img)
        number += 1


if __name__ == '__main__':

    quit = False
    queue = multiprocessing.Queue()
    multiprocessing.Process(target=save, args=(queue,)).start()

    while not quit:
        img = screenshot()       
        queue.put(img)

        cv2.imshow("Record Screen", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            quit = True
            queue.put(None)
            

