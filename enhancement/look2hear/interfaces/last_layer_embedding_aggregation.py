import torch


class LastLayerAggr(torch.nn.Module):
    def __init__(self, downstream_timsteps=256):
        super().__init__()
        self.downstream_timsteps = downstream_timsteps

    def forward(self, x):
        z = torch.nn.functional.interpolate(x, size=(self.downstream_timsteps, x.shape[-1]), mode='bilinear', align_corners=False)
        
        return z[:, -1, :, :].permute(0, 2, 1).contiguous()