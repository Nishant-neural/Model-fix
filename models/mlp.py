import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)      
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x