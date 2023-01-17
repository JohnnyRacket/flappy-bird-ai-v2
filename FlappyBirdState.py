import time
import torch
import torchvision.transforms as transforms

from actions import press_spacebar
from capture_screen import screenshot
from action_space import Actions
from gamestate_classifier import gamestate_classifier
from filter_images import gamestate_classifier_filter, rl_agent_filter

class FlappyBirdState(object):
    def __init__(self):
        self.done = False
        self.timeAlive = 0
        self.bbox = {'top': 92, 'left': 186, 'width': 320, 'height': 400}
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        self.img_seq = []
        self.total_reward = 0
        self.total_steps = 0

    def step(self, action):

        reward = 10
        if(action == 1):
            press_spacebar()
            reward = -10
        
        img = screenshot(self.bbox)
        
        img2 = gamestate_classifier_filter(img)
        gamestate = gamestate_classifier.classify_gamestate(img2)

        img = rl_agent_filter(img)
        img = self.transform(img)
        # reward for every step alive
        

        # need to check for terminal
        if (gamestate == 'game_over'):
            self.done = True
            reward = -1000
        # print(img.shape)
        self.img_seq.pop(0)
        self.img_seq.append(img)
        img_seq = torch.cat((self.img_seq[0],self.img_seq[1],self.img_seq[2],self.img_seq[3]))
        self.total_reward += reward
        self.total_steps += 1
        return img_seq, reward, self.done

    def reset(self, first = False):
        time.sleep(.5)
        press_spacebar()
        if not first:
            time.sleep(1)
            press_spacebar()
        self.done = False
        print(self.total_reward)
        print(self.total_steps)
        print()
        self.total_reward = 0
        self.total_steps = 0
        img = screenshot(self.bbox)
        img = rl_agent_filter(img)
        img = self.transform(img)
        for e in range(4):
            self.img_seq.append(img)
        return torch.cat((img,img,img,img))


class SkipFrame():
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        self.env = env
        self._skip = skip

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