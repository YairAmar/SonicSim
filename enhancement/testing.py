import torch
import hydra
from look2hear.interfaces import FeatureFusionInterface
from look2hear.upstream_encoders import WavLMWrapper
from look2hear.models.fullsubnet_wavlm import FullSubnetWavLM
if __name__ == "__main__":
    # Example usage
    # model = FeatureFusionInterface(downstream_feature_space=640,
    #                                downstream_timesteps=256,
    #                                upstream_feature_space=1024,
    #                                fusion_type='concat',
    #                                aggregation_type='rfilm',
    #                                fusion_residual=True,
    #                                fusion_use_conv=True,
    #                                aggregation_layer_wise=True)
    x = torch.randn(7, 16000*4)  # Example upstream feature (batch_size=1, sequence_length=16000)
    # downstream_x = torch.randn(7, 640, 256)  # Example downstream feature (batch_size=1, feature_dim=256)
    # upstream_model = WavLMWrapper()
    # upstream_x = upstream_model(x)
    # output = model(upstream_x, downstream_x)
    hydra.initialize(config_path="./config", job_name="test_app")
    cfg = hydra.compose(config_name="fullsubnet_wavlm")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    output = model(x)  # Using the same tensor for both upstream and downstream for testing
    # Print the shape of the output
    print(output[0].shape)  # Should be (7, 640, 256) for downstream feature
    print(output[1].shape)  # Should be (7, 640, 256) for downstream feature
    print(output[2].shape)  # Should be (7, 640, 256) for downstream feature