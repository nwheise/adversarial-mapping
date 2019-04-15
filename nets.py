import torch.nn as nn
import torch.nn.functional as F

class MappingNet(nn.Module):
    
    def __init__(self):
        super(MappingNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=2)
    

    def forward(self, x):
        x = self.fc1(x)
        return x


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x