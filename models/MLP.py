import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim:int, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.norm1(self.fc1(x)))
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.norm2(self.fc2(x1)))
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.fc3(x2))
        return torch.sigmoid(self.out(x3))
