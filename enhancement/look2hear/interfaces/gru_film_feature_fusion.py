import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUFiLMFeatureFusion(nn.Module):
    """
    GRU-based FiLM Feature Fusion
    """
    def __init__(self, 
                 upstream_feature_space: int = 1024, 
                 downstream_feature_space: int = 256,
                 residual: bool = True,
                 use_conv: bool = False):
        """
        Initialize the GRU FiLM Feature Fusion module.
        Args:
            upstream_feature_space: The feature space of the upstream model.
            downstream_feature_space: The feature space of the downstream model.
            residual: Whether to use residual connections.
            use_conv: Whether to add convolutional layers before the GRU.
        """
        super().__init__()
        self.upstream_feature_space = upstream_feature_space
        self.downstream_feature_space = downstream_feature_space
        self.residual = residual
        self.use_conv = use_conv

        # Optional convolutional layers
        if self.use_conv:
            self.conv1 = nn.Conv2d(upstream_feature_space, upstream_feature_space, kernel_size=(3, 3), padding=(1, 1))
            self.conv2 = nn.Conv2d(upstream_feature_space, upstream_feature_space, kernel_size=(3, 3), padding=(1, 1))
            self.conv_activation = nn.ReLU()

        # GRU layers
        self.gru = nn.GRU(
            input_size=upstream_feature_space,
            hidden_size=downstream_feature_space,
            num_layers=2,
            batch_first=True
        )

        # Linear projection for alpha and beta
        self.proj = nn.Linear(downstream_feature_space, downstream_feature_space * 2)

    def forward(self, x_upstream: torch.Tensor, x_downstream: torch.Tensor) -> torch.Tensor:
        """
        Fuse the two features by applying GRU-based FiLM over the downstream feature space.
        Args:
            x_upstream: The first feature tensor.
            x_downstream: The second feature tensor.
        """
        # Apply optional convolutional layers
        if self.use_conv:
            x_upstream = self.conv_activation(self.conv1(x_upstream))
            x_upstream = self.conv_activation(self.conv2(x_upstream))

        # Prepare input for GRU
        b, c, h, w = x_upstream.shape
        x_upstream = x_upstream.view(b, c, -1).permute(0, 2, 1)  # [B, T, C]

        # GRU forward pass
        gru_out, _ = self.gru(x_upstream)  # [B, T, downstream_feature_space]

        # Linear projection to compute alpha and beta
        z = self.proj(gru_out)  # [B, T, downstream_feature_space * 2]
        alpha = z[..., :self.downstream_feature_space]
        alpha = F.softmax(alpha, dim=2)
        beta = torch.tanh(z[..., self.downstream_feature_space:])

        # Apply FiLM modulation
        x_downstream = x_downstream.view(b, c, -1)  # [B, C, T]
        modulated_features = x_downstream * alpha.permute(0, 2, 1) + beta.permute(0, 2, 1)

        if self.residual:
            modulated_features += x_downstream

        return modulated_features.view(b, c, h, w)