import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

#loads the CIFAR10 dataset from the ./data folder into the the array trainset. This is 50000x1 array of tuples. The tuples contain a 3x32x32 array to represent the 3 channels of a 32x32 color image
#, and a number 1-10 to represent the category of the image. The trainloader wraps the trainset in an iterable, and the same applies for the testset and testloader.

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#a network based on cnn explainer that didn't train very well

#class NeuralNetwork(nn.Module):
#    def __init__(self):
#        super(NeuralNetwork, self).__init__()
#        self.linear_cnn_stack = nn.Sequential(
#            nn.Conv2d(3,10,3),
#            nn.ReLU(),
#            nn.Conv2d(10,10,3),
#            nn.ReLU(),
#            nn.MaxPool2d(2,2),
#            nn.Conv2d(10,10,3),
#            nn.ReLU(),
#            nn.Conv2d(10,10,3),
#            nn.ReLU(),
#            nn.MaxPool2d(2,2),
#            nn.Flatten(),
#            nn.Linear(250,10),
#            nn.Softmax()
#        )
#
#    def forward(self, x):
#        logits = self.linear_cnn_stack(x)
#        return logits

#the example network in the pytorch documentation that trained very well (loss from 2.304 -> 1.167 in 10 epochs)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#self explanatory

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss=0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("done")
