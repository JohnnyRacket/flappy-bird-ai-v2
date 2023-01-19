import time
import torch
import torchvision.transforms as transforms
import numpy as np

from actions import press_spacebar
from capture_screen import screenshot
from action_space import Actions
from gamestate_classifier import gamestate_classifier
from crash_classifier import crash_classifier
from filter_images import gamestate_classifier_filter, rl_agent_filter, penalty_classifier_filter

class FlappyBirdState(object):
    def __init__(self):
        self.done = False
        self.step_time = 0
        self.bbox = {'top': 147, 'left': 196, 'width': 320, 'height': 420}
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        self.img_seq = []
        self.total_reward = 0
        self.total_steps = 0
        x = np.linspace(-1, 1, 80).astype(np.float16)
        y = np.linspace(-1, 1, 105).astype(np.float16)
        xv, yv = np.meshgrid(x, y)
        self.xv = torch.tensor(xv).unsqueeze(0)
        self.yv = torch.tensor(yv).unsqueeze(0)

    def step(self, action):

        if(action == 1):
            press_spacebar()
        
        img = screenshot(self.bbox)
        
        img2 = gamestate_classifier_filter(img[15:-5,:])
        gamestate = gamestate_classifier.classify_gamestate(img2)

        img3 = penalty_classifier_filter(img)
        penalty = crash_classifier.classify_gamestate(img3)

        img = rl_agent_filter(img)
        img = self.transform(img)
        # reward for every step alive
        


        # print(img.shape)
        self.img_seq.pop(0)
        self.img_seq.append(img)
        img_seq = torch.cat((self.img_seq[0],self.img_seq[1],self.img_seq[2],self.img_seq[3], self.xv, self.yv))

        # reward = (time.time() - self.step_time) * 34
        reward = 2
        # need to check for terminal
        

        if(penalty == 'penalty'):
            reward = -2
            if (action == 1):
                reward = -4

        if (gamestate == 'game_over'):
            self.done = True
            reward = -100

        self.total_reward += reward
        self.total_steps += 1
        self.step_time = time.time()
        return img_seq, reward, self.done

    def reset(self, first = False):
        img = screenshot(self.bbox)
        
        img2 = gamestate_classifier_filter(img[20:,:])
        gamestate = gamestate_classifier.classify_gamestate(img2)

        
        self.done = False
        print('reward = ', self.total_reward, ', steps = ', self.total_steps)
        # # print(self.total_steps)
        # print()
        self.total_reward = 0
        self.total_steps = 0
        img = screenshot(self.bbox)
        img = rl_agent_filter(img)
        img = self.transform(img)
        for e in range(4):
            self.img_seq.append(img)
        self.step_time = time.time()

        if(gamestate == 'game_over'):
            time.sleep(.5)
            press_spacebar()
            if not first:
                time.sleep(1.5)
                press_spacebar()
        elif(gamestate == 'title_screen'):
            time.sleep(1)
            press_spacebar()
            print('title screen recovery')
        elif(gamestate == 'playing'):
            time.sleep(3)
            press_spacebar()
            time.sleep(1.5)
            press_spacebar()
            print('playing screen recovery')

        return torch.cat((img,img,img,img,self.xv, self.yv))


class SkipFrame():
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        self.env = env
        self._skip = skip
        self.total_reward = self.env.total_reward
        self.total_steps = self.env.total_steps

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            img, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
        return img, total_reward, done
    def reset(self, first = False):
        return self.env.reset(first)