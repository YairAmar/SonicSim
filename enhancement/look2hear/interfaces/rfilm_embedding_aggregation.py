import torch
import torch.nn as nn
import torch.nn.functional as F


class RFiLMEmbeddingAggregation(torch.nn.Module):
    def __init__(self, layer_wise=True):
        super().__init__()
        self.gru = GRUProjector()
        self.conv_backbone = SeparableConvReduceHeight(channels=1024)

    def forward(self, x):
        z = torch.nn.functional.interpolate(x, size=256, mode='linear', align_corners=False)
        z = self.conv_backbone(z)
        alpha, beta = self.gru(z)

        return (torch.sum((x * alpha), dim=2) + beta)


class SeparableConvReduceHeight(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        layers = []
        in_h = 24
        for _ in range(5):
            # Choose kernel size and stride to reduce height progressively
            k = 3
            s = 2 if in_h > 1 else 1
            pad = 1

            layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(k, 12),  # Vertical only
                    stride=(s, 6),
                    padding=(pad, 0),
                    groups=channels,  # Depthwise
                    bias=False
                ),
                nn.ReLU()
            ))
            in_h = (in_h + 2 * pad - k) // s + 1  # Update height
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRUProjector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=25, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # set True if you want bi-GRU
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, 1024, 1, 256] → squeeze → [B, 1024, 256]
        x = x.squeeze(2)  # remove the 1-height dim → [B, 1024, 256]
        b, f, t = x.shape
        x = x.permute(0, 2, 1)  # [B, 256, 1024]
       
        # GRU
        gru_out, _ = self.gru(x)  # [B, 256, hidden_dim]
       
        # Linear projection + activation
        out = self.proj(gru_out)  # [B, 256, 25]
        alpha = out[..., :24]
        beta = F.tanh(out[..., -1])
        alpha = F.softmax(alpha, dim=2)
        
        return alpha.reshape(b, 1, 24, t), beta.reshape(b, 1, t)