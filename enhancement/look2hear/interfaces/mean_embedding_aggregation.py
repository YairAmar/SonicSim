import torch


class MeanAggr(torch.nn.Module):
    def __init__(self, downstream_time_steps=256):
        super().__init__()
        self.downstream_time_steps = downstream_time_steps

    def forward(self, x):
        print(x.shape)
        # Ixnterpolate only over the x.shape[2] dimension
        z = torch.nn.functional.interpolate(
            x,size=(self.downstream_time_steps, x.shape[3]), mode='bilinear', align_corners=False
        )
        return z.mean(dim=1).permute(0, 2, 1).contiguous()