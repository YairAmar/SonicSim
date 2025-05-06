import torch


class LastLayerAggr(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.nn.functional.interpolate(x, size=256, mode='linear', align_corners=False)
        
        return z[..., -1, :]