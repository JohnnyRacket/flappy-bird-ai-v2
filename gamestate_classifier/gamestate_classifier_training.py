
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from GamestateDataset import GamestateClassifierDataset
from GamestateNet import GamestateClassifierNet

# use gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = GamestateClassifierDataset(csv_file='flappy_bird_gamestate_labels.csv', root_dir = 'gamestate_classifier_imgs', transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]))
print(len(dataset))
in_channel = 1
num_classes = 4
learning_rate = 1e-3
momentum = 0.9
batch_size = 32
num_epochs = 500


trainset, testset = torch.utils.data.random_split(dataset, [480, 121])
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True) 


classes = ('title_screen', 'playing', 'game_over')

gamestateClassifierNet = GamestateClassifierNet()
gamestateClassifierNet.to(device)

# cross entropy + softmax = mutually exclusive

# loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(gamestateClassifierNet.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    
    print(epoch)

    running_loss = 0.0
    for i, (data, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        data = data.to(device=device)
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = gamestateClassifierNet(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# save net
PATH = './gamestate_classifier_net.pth'
torch.save(gamestateClassifierNet.state_dict(), PATH)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = GamestateClassifierNet()
net.to(device)
net.load_state_dict(torch.load(PATH))
net.eval()   

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for (data, labels) in testloader:
        images = data.to(device=device)
        labels = labels.to(device=device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total} %')
