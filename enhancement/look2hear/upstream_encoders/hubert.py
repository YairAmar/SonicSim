import torch
from transformers import HubertModel
from .abs_upstream import AbstractUpstreamEncoder


class HuBERTWrapper(AbstractUpstreamEncoder):
    def __init__(self, model_path: str = None):
        super().__init__()
        model_path = model_path if model_path else "facebook/hubert-large-ls960-ft"
        self.model = HubertModel.from_pretrained(model_path)
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass for HuBERT.

        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Hidden states from the HuBERT model.
        """
        with torch.no_grad():
            # Pass the audio signal through the WavLM model
            outputs = self.model(x, output_hidden_states=True)
        
        # Extract hidden states from all transformer layers
        hidden_states = outputs.hidden_states  # Tuple of (layer_0, layer_1, ..., layer_23)
        
        # Convert tuple to list for easier manipulation
        # Stack hidden states along a new dimension (layer dimension)
        return torch.stack(hidden_states, dim=1)