import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class tensorWalk:
    def __init__(self, data):
        self.data = data
        self.x_data = torch.tensor(data)

    def printTensors(self):

        #print(self.data, self.x_data)
        x_test = torch.rand(50,32,32,3)
        #print(x_test[0].shape)
        x_train = torch.ones(4,4,4)
        x_train[:,:,1] = 0
        print(x_train)

        print(f"Shape of tensor: {x_train.shape}")
        print(f"Datatype of tensor: {x_train.dtype}")
        print(f"Device tensor is stored on: {x_train.device}")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":

    tw = tensorWalk([[1,2],[3,4]])
    tw.printTensors()

    #related to the networks
    network = Net()
    print(network)

    params = list(network.parameters())
    print(len(params))
    print(params[0].size())

    input = torch.randn(1, 1, 32, 32)
    out = network(input)
    print(out)

