import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatinationFeatureFusion(torch.nn.Module):
    """
    Concatenation Feature Fusion
    """
    def __init__(self, 
                 upstream_feature_space: int = 1024, 
                 donwnstream_feature_space: int = 256):
        super().__init__()
        self.linear = nn.Linear(upstream_feature_space + donwnstream_feature_space, 
                                donwnstream_feature_space)

    def forward(self, upstream_x, downstream_x):
        """
        Fuse the two features by concatenating them and applying a linear layer.
        Args:
            x1: The first feature tensor.
            x2: The second feature tensor.
        """
        z = torch.concatenate((upstream_x, downstream_x), dim=1)
        return self.linear(z)
    