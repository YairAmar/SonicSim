import torch


class ConcatinationFeatureFusion(torch.nn.Module):
    """
    Concatenation Feature Fusion
    """
    def __init__(self, 
                 upstream_feature_space: int = 1024, 
                 donwnstream_feature_space: int = 256):
        super().__init__()
        self.linear = torch.nn.Linear(upstream_feature_space + donwnstream_feature_space, 
                                donwnstream_feature_space)

    def forward(self, x_upstream, x_downstream):
        """
        Fuse the two features by concatenating them and applying a linear layer.
        Args:
            x_upstream: The first feature tensor.
            x_downstream: The second feature tensor.
        """
        z = torch.concatenate((x_upstream, x_downstream), dim=1)
        z = self.linear(z.permute(0, 2, 1))
        return z.permute(0, 2, 1)
    