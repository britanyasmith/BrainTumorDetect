import torch 
import torch.nn as nn #defining the neural network 
import torch.nn.functional as F #import convolution functions like ReLu
import torch.optim as optim #import the optimizer (such as SGD or adam, etc)











class Net(nn.Module): 
    '''Description: Convolution Neural Network '''

    def __init__(self): 
        '''Initializing the Network'''
        super().__init__()
        # 3 input image channel, 6 ouput channels,
        # 5x5 square convolution kernel 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels= 16, kernel_size=5) 
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  #Flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)