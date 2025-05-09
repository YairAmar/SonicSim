import torch

from ..interfaces.concatination_feature_fusion import ConcatinationFeatureFusion
from ..interfaces.gru_film_feature_fusion import GRUFiLMFeatureFusion
from ..interfaces.linear_film_feature_fusion import LinearFiLMFeatureFusion
from ..interfaces.mean_embedding_aggregation import MeanAggr
from ..interfaces.last_layer_embedding_aggregation import LastLayerAggr
from ..interfaces.rfilm_embedding_aggregation import RFiLMEmbeddingAggregation


class FeatureFusionInterface(torch.nn.Module):
    """
    Feature Fusion Interface
    """
    def __init__(self, 
                 aggregation_type: str = 'mean', 
                 fusion_type: str = 'concat',
                 upstream_feature_space: int = 1024, 
                 downstream_feature_space: int = 256,
                 downstream_timesteps: int = 256,
                 fusion_residual: bool = True,
                 fusion_use_conv: bool = True,
                 aggregation_layer_wise: bool = True,
                 ):
        """
        Initialize the Feature Fusion Interface module.
        Args:
            aggregation_type: The type of aggregation to use.
            fusion_type: The type of fusion to use.
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.aggregation_type = aggregation_type

        if aggregation_type == 'mean':
            self.aggregation = MeanAggr(downstream_time_steps=downstream_timesteps)
        elif aggregation_type == 'last_layer':
            self.aggregation = LastLayerAggr(downstream_timsteps=downstream_timesteps)
        elif aggregation_type == 'rfilm':
            self.aggregation = RFiLMEmbeddingAggregation(layer_wise=aggregation_layer_wise, downstream_time_steps=downstream_timesteps)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
        
        # Initialize the fusion method based on the specified type

        if fusion_type == 'concat':
            self.fusion = ConcatinationFeatureFusion(upstream_feature_space=upstream_feature_space, 
                                                    donwnstream_feature_space=downstream_feature_space)
        elif fusion_type == 'gru_film':
            self.fusion = GRUFiLMFeatureFusion(upstream_feature_space=upstream_feature_space,
                                               downstream_feature_space=downstream_feature_space,
                                               residual=fusion_residual,
                                               use_conv=fusion_use_conv)
        elif fusion_type == 'linear_film':
            self.fusion = LinearFiLMFeatureFusion(upstream_feature_space=upstream_feature_space,
                                                 downstream_feature_space=downstream_feature_space,
                                                 residual=fusion_residual)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, 
                upstream_x: torch.Tensor, 
                downstream_x: torch.Tensor) -> torch.Tensor:
        z = self.aggregation(upstream_x)
        z = self.fusion(z, downstream_x)
        
        return z
