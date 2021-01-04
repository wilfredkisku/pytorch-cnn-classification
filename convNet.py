import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

#hyper parameters
learning_rate = 0.001
momentum = 0.9
epochs = 10
batch_size = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.BN1 = nn.BatchNorm2D(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.BN2 = nn.BatchNorm2D(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #########################DIFFERENT ACTIVATIONS FOR THE CONV LAYERS########################
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.sigmoid(self.BN1(self.conv1(x))))
        #x = self.pool(F.tanh(self.BN1(self.conv1(x))))
        #x = self.pool(nn.Identity(self.BN1(self.conv1(x))))
        
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.sigmoid(self.BN2((self.conv2(x))))
        #x = self.pool(F.tanh(self.BN2(self.conv2(x))))
        #x = self.pool(nn.Identity(self.BN2(self.conv1(x))))

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

################PRINT RANDOM TRAIN SAMPLE FOR DEBUGGING##########################
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()

############LOSSES###########################
criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
#criterion = nn.MSELoss()

############OPTIMIZERS########################
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


for epoch in range(epochs):
    run_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'%(epoch + 1, i + 1, run_loss / 2000))
            run_loss = 0.0

print('Finished Training ...')

PATH = 'data/cifar_net.pth'
torch.save(net.state_dict(), PATH)

#####################PRINT RANDOM TEST SAMPLES FOR DBUGGING#####################
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
