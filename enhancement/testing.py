import torch
import hydra
from look2hear.interfaces import FeatureFusionInterface
from look2hear.upstream_encoders import WavLMWrapper
from look2hear.models.mpsenet import MPSENet

if __name__ == "__main__":
    # Load configuration
    with hydra.initialize(config_path="config"):
        cfg = hydra.compose(config_name="mpsenet.yaml")

    # Initialize the model
    model = hydra.utils.instantiate(cfg.model)
    # model = MPSENet(**cfg.model)
    model.eval()

    # Create a dummy input signal (batch_size=1, channels=1, length=16000)
    dummy_signal = torch.randn(2, 64000)

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_signal)

    # Print the output shape
    print("Forward pass successful with output shape: ", output[3].shape)

