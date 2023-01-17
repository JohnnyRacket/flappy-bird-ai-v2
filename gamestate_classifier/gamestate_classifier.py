import torch
import torchvision.transforms as transforms
import numpy as np
from gamestate_classifier.GamestateNet import GamestateClassifierNet

PATH = './gamestate_classifier/gamestate_classifier_net.pth'
classes = ('title_screen', 'playing', 'game_over')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = np.linspace(0, 255, 640)
y = np.linspace(0, 255, 280)
xv, yv = np.meshgrid(x, y)

net = GamestateClassifierNet()
net.to(device)
net.load_state_dict(torch.load(PATH))
net.eval()   

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

def classify_gamestate(img):
    # conv_coord = np.dstack((img, xv, yv)).astype(np.uint8)
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
           # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
        class_index = net(tensor).argmax()
        prediction = classes[class_index]
        return prediction
