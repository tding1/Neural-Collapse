import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden, depth = 6, fc_bias = True, num_classes = 10):
        # Depth means how many layers before final linear layer
        
        super(MLP, self).__init__()
        layers = [nn.Linear(3072, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, num_classes, bias = fc_bias)
        print(fc_bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        x = self.fc(x)
        return x, features
