import torch

import torch.nn as nn
import torch.nn.functional as F

class convModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv2D(in_channels, out_channels, kernel_size, ...)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.batchnorm5 = nn.BatchNorm2d(128)
        
        # MaxPool2D(kernel_size, stride, ...)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.4)
        
        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 84)
        self.fc4 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))               # 32 filtros
        x = F.relu(self.batchnorm2(self.conv2(x)))               # 64 filtros
        x = self.maxpool(self.batchnorm3(self.conv3(x)))        # 128 filtros
        x = F.relu(x)

        x = F.relu(self.batchnorm4(self.conv4(x)))               # 128 filtros
        x = self.maxpool(self.batchnorm5(self.conv5(x)))        # 128 filtros
        x = F.relu(x)
        
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)            # 40%

        x = F.relu(self.fc2(x))
        x = self.dropout1(x)

        x = F.relu(self.fc3(x))
        x = self.dropout1(x)

        x = self.fc4(x)
        
        return x
