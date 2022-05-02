import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import numpy as np


BATCH_SIZE = 32

## transformations
transform = transforms.Compose([transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
""" Data loader is a wrapper around the data set which you can index to"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

## download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

print("Data train len: " , len(trainset))
#print(trainset[10])


####################### Model #######################
'''
Note that MyModel makes uses of the predefined modules Conv2d and Linear, which
 it instantiates in its constructor. Running data x through a module conv1 simply
 consists of calling it like a function: out = conv1(x).
'''

class MyModel(nn.Module):
    def __init__(self):            # Init and main parameters
        super(MyModel, self).__init__()

        # 28x28x1 => 26x26x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # Convolution
        self.d1 = nn.Linear(26 * 26 * 32, 128)                                # Linear layer (input chanel, output chanel)
        self.d2 = nn.Linear(128, 10)                                          # Linear layer. 10 for number classification             
        
    def forward(self, x):          # Computation graph from the input to the output
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)
        #x = x.view(32, -1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out
  
# Make a tensor from np array
a = np.array([[1,2],[3,4]])
b = np.ones((2,2))
ta = torch.tensor(a, dtype=float)
tb = torch.tensor(b, dtype = float)

print(torch.matmul(ta, tb))
print(ta @ tb)

print(torch.cuda.is_available())


learning_rate = 0.001
num_epochs = 2 #5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss() #loss 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optimizer

####################### Training #######################
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    ## training step
    """ for an iteration we extract the number of examples equal to a batch size """
    for i, (images, labels) in enumerate(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        logits = model(images)
        loss = criterion(logits, labels)
        
        # set grad to 0 (pytorch usually accumulates the grad)
        optimizer.zero_grad() 
        loss.backward()

        ## update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += (torch.argmax(logits, 1).flatten() == labels).type(torch.float).mean().item()
    
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i))
        
####################### Testing #######################
test_acc = 0.0
for i, (images, labels) in enumerate(testloader, 0):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_acc += (torch.argmax(outputs, 1).flatten() == labels).type(torch.float).mean().item()
    preds = torch.argmax(outputs, 1).flatten().cpu().numpy()
    print(labels)
    print(preds)
        
print('Test Accuracy: %.2f'%(test_acc/i))
#Test Accuracy: 0.98
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
#        [1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6]