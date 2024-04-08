import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision

class MyCNN(nn.Module):
    def __init__(self,drop = 0.2):
        super(MyCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
            nn.Dropout(),  
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  
            nn.Dropout(),  
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # max_pooling2d_7
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(256 * 3 * 3, 256),  # dense_2
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),  
            nn.Linear(256, 1024),  
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 7)  
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
