import torch
import torchvision.transforms as transforms
import numpy as np
from crash_classifier.CrashNet import CrashClassifierNet

PATH = './crash_classifier/crash_classifier_net.pth'
classes = ('safe', 'penalty')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# conv_coord = np.dstack((img, xv, yv)).astype(np.uint8)
x = np.linspace(0, 1, 18).astype(np.float16)
y = np.linspace(0, 1, 210).astype(np.float16)
xv, yv = np.meshgrid(x, y)
yv_tensor = torch.tensor(yv).unsqueeze(0)

net = CrashClassifierNet()
net.to(device)
net.load_state_dict(torch.load(PATH))
net.eval()   

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

def classify_gamestate(img):
    
    tensor = transform(img)
    tensor = torch.cat((tensor, yv_tensor))
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
           # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
        class_index = net(tensor).argmax()
        prediction = classes[class_index]
        return prediction
