import torch
import torch.nn.functional as F

class LinearFiLMFeatureFusion(torch.nn.Module):
    """
    FiLM Feature Fusion
    """
    def __init__(self, 
                 upstream_feature_space: int = 1024, 
                 downstream_feature_space: int = 256,
                 residual: bool = True):
        """
        Initialize the Linear FiLM Feature Fusion module.
        Args:
            upstream_feature_space: The feature space of the upstream model.
            downstream_feature_space: The feature space of the downstream model.
            residual: Whether to use residual connections.
        """
        super().__init__()
        self.upstream_feature_space = upstream_feature_space
        self.downstream_feature_space = downstream_feature_space
        self.residual = residual
        self.linear1 = torch.nn.Linear(upstream_feature_space, upstream_feature_space)
        self.linear2 = torch.nn.Linear(upstream_feature_space, downstream_feature_space * 2) 

    def forward(self, 
                x_upstream: torch.Tensor, 
                x_downstream: torch.Tensor) -> torch.Tensor:
        """
        Fuse the two features by applying Residual FiLM over the downstream feature space.
        Args:
            x_upstream: The first feature tensor.
            x_downstream: The second feature tensor.
        """
        z = self.linear1(x_upstream)
        z = F.relu(z)
        z = self.linear2(z)

        alpha = z[..., :self.downstream_feature_space]
        alpha = F.softmax(alpha, dim=2)
        beta = z[..., self.downstream_feature_space:]
        
        if self.residual:
            modulated_features = x_downstream * alpha + beta + x_downstream
        else:
            modulated_features = x_downstream * alpha + beta
            
        return modulated_features
