import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):

    def __init__(self, input_dims, output_dim: int = 128):
        super().__init__()
        assert len(input_dims) == 3
        # Layer 1
        self.c1 = nn.Conv2d(input_dims[0], 32, 3, 1)
        self.b1 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(p=0.1)
        self.m1 = nn.MaxPool2d(2, 2)
        # Layer 2
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.b2 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(p=0.1)
        self.m2 = nn.MaxPool2d(2, 2)
        # Layer 3
        self.c3 = nn.Conv2d(64, 64, 5, 1)
        self.b3 = nn.BatchNorm2d(64)
        self.d3 = nn.Dropout2d(p=0.1)
        self.m3 = nn.MaxPool2d(2, 2)
        # Linear layers
        self.flatten = nn.Flatten()  # Dimension 8704x1
        self.l1 = nn.Linear(8704, 256)
        self.b4 = nn.BatchNorm1d(256)
        self.d4 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.m1(self.d1(self.b1(F.relu(self.c1(x)))))
        x = self.m2(self.d2(self.b2(F.relu(self.c2(x)))))
        x = self.m3(self.d3(self.b3(F.relu(self.c3(x)))))
        x = self.flatten(x)
        x = self.d4(self.b4(F.relu(self.l1(x))))
        x = self.l2(x)
        return x
