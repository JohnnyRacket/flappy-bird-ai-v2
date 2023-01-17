import win32gui
import win32con
import cv2
import time
import webbrowser
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import pathlib
import datetime
from actions import press_spacebar
from FlappyBirdState import FlappyBirdState, SkipFrame
from FlappyBirdAgent import Bird
from capture_screen import screenshot

from gamestate_classifier import gamestate_classifier
from filter_images import gamestate_classifier_filter, rl_agent_filter
from rl_agent import BirdNet


####################################### UI SETUP ##############################################
print('start training')
webbrowser.get('C:/Program Files/Google/Chrome/Application/chrome.exe %s').open('https://flappybird.ee/old')
hwnd = win32gui.FindWindow(None, "Play Flappy Bird Online Old - Google Chrome")
win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
win32gui.MoveWindow(hwnd, -8, 0, 700, 616, True)
win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
win32gui.SetForegroundWindow(hwnd)
bbox = win32gui.GetWindowRect(hwnd)
game_bbox = {"top": bbox[0] + 100, "left": bbox[1] + 186, "width": bbox[2] - 372, "height": bbox[3] - 216}
#################################################################################################


##################################### TRAINING SETUP ############################################

# x = np.linspace(0, 255, 80).astype(np.uint8)
# y = np.linspace(0, 255, 100).astype(np.uint8)
# xv, yv = np.meshgrid(x, y)

#################################################################################################


############################################ LOOP ###############################################
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = pathlib.Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

env = FlappyBirdState()
env = SkipFrame(env, 4)

bird = Bird(state_dim=(4, 400, 320), action_dim=2, save_dir=save_dir)

# logger = MetricLogger(save_dir)
episodes = 50000
for e in range(episodes):
    if(e == 0):
        state = env.reset(True)
        print('first')
    else:
        state = env.reset()
        # print('reset env')
    print(e)

    # Play the game!
    while True:
        start = time.time()
        # Run agent on the state
        action = bird.act(state)

        # Agent performs action
        next_state, reward, done = env.step(action)

        # Remember
        # print(state.shape, next_state.shape)
        bird.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = bird.learn()

        # Logging
        # logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break        
        # ... do stuff that might take significant time
        time.sleep(max(1./10 - (time.time() - start), 0))

    # logger.log_episode()

    # if e % 20 == 0:
    #     logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)