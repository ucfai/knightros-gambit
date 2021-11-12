
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data.dataloader import DataLoader


# Tested and somewhat optimized hyperparameters
epochs = 8
batch_size = 4
lr = 0.028


transform = ToTensor()

train_set = datasets.MNIST(root='data', train=True, transform=transform)
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = datasets.MNIST(root='data', train=False, transform=transform)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)


# returns accuracy of predictions to the actual label
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class MnistConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


model = MnistConvNet()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=lr)

loss_sum = 0
acc_sum = 0

# Training
for epoch in range(epochs):
    for img, label in train_dl:
        # Make predictions with model, calc loss using Cross Entropy and turn them to %s w/ softmax
        # (softmax is part of the CrossEntropyLoss function)
        preds = model(img)
        loss = loss_fn(preds, label)
        loss_sum += loss.item()
        acc = accuracy(preds, label)
        acc_sum += acc
        
        #Calculate gradients
        loss.backward()
        #Update parameters
        opt.step()
        #Reset gradients
        opt.zero_grad()

print("Training Loss:{:.10f}  Training Accuracy:{:.10f}".format(loss_sum/(epochs*len(train_dl)), acc_sum/(epochs*len(train_dl))))


loss_sum = 0
acc_sum = 0

# Testing
for img, label in val_dl:
        preds = model(img)
        loss = loss_fn(preds, label)
        loss_sum += loss.item()
        acc = accuracy(preds, label)
        acc_sum += acc

print("Loss:{:.10f}  Accuracy:{:.10f}".format(loss_sum/(len(val_dl)), acc_sum/(len(val_dl))))

