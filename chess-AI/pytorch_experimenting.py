import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

device = "cpu" # can choose device with pytorch, using cpu for now

class NeuralNetwork(nn.Module):
    
    def __init__(self):

        super(NeuralNetwork,self).__init__()

        self.flatten = nn.Flatten() ## takes input and turns into a 1D tensor

        self.linear_relu_stack = nn.Sequential( ## the sequential layers
            nn.Linear(28*28,512), ## 512 input and output nodes
            nn.ReLU(), ## relue activation function
            nn.Linear(512,512), ##512 input nodes and output
            nn.ReLU(),
            nn.Linear(512,10) ## needs to have output layer with 10 because of 10 different classes
        )

    def forward(self,x): ## runs the foward apss of the data
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train() ## specifies that the model is in train mode

    for batch, (X,y) in enumerate(dataloader):

        X,y = X.to(device), y.to(device)

        pred = model(X) # sees what the model predicts on x
        loss = loss_fn(pred,y) # uses the loss function

        
        ##### BACK PROP STEPS ####

        optimizer.zero_grad() ## sets all the gradients to zero
        loss.backward() ## accumalates the graients for each parameter
        optimizer.step() ## performs paramter updates with current gradients

        ###########################

        if ( batch %100 == 0) :
            loss,current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
 

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # specifies model is in test mode
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


## GETS THE TEST AND TRAIN DATA FROM MNIST 

train_data = datasets.FashionMNIST(
   root="data",
   train=True,
   download=True,
   transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
   root="data",
   train=False,
   download=True,
   transform=ToTensor(),
)

########################################

# Data Set stores samples and their labels, DataLoader is an iterable around the dataset
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
model = NeuralNetwork().to(device)

## LOSS AND OPTIMIZER 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

######################

epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

####### TRY OUT MODEL ########

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

x, y = test_data[0][0], test_data[0][1] ## uses the first entry of the test data
pred = model(x) ## gets tensor of different predictions
print(pred)
prediction = pred[0].argmax(0) ## argmax used to find index with greates value
print(f'Prediction :" {classes[prediction]}", Actual: "{classes[y]}"')


################################