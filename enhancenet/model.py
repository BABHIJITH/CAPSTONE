import torch
import torch.nn as nn

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 256 * 256, 3 * 256 * 256)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        x = torch.sigmoid(self.fc(x))  
        return x.view(-1, 3, 256, 256)  
