import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNet(nn.Module):

    def __init__(self):
        super(MappingNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=4)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)

        return x


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return torch.sigmoid(x)
