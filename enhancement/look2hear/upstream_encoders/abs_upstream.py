from abc import ABC, abstractmethod
import torch

class AbstractUpstreamEncoder(ABC, torch.nn.Module):
    """
    Abstract class for upstream encoders.
    All upstream encoders should inherit from this class and implement the required methods.
    """

    def __init__(self, model_path: str = None):
        super().__init__()
        
    @abstractmethod
    def forward(self, input_values):
        """
        Forward pass for the upstream encoder.

        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Encoded representations from the upstream encoder.
        """
        pass
