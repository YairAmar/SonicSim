import torch
from transformers import WavLMModel
from .abs_upstream import AbstractUpstreamEncoder


class WavLMWrapper(AbstractUpstreamEncoder):
    def __init__(self, model_path: str = None):
        """
        Wrapper class for using a WavLM model from Hugging Face.
        
        Args:
            model_path (str, optional): Path to the locally saved WavLM model. If None, 
                                        the model will be loaded from Hugging Face.
        """
        super(WavLMWrapper, self).__init__()
        model_path = model_path if model_path else "microsoft/wavlm-large"
        self.model = WavLMModel.from_pretrained(model_path)
        
        # Freeze WavLM weights
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, sampling_rate: int = 16000):
        """
        Forward pass for the WavLM model.
        
        Args:
            audio_signal (torch.Tensor): Input audio signal (batch_size, sequence_length).
            sampling_rate (int): Sampling rate of the input audio signal. Default is 16000.
        
        Returns:
            List[torch.Tensor]: A list of activations from all 24 transformer layers.
        """
        with torch.no_grad():
            # Pass the audio signal through the WavLM model
            outputs = self.model(x, output_hidden_states=True)
        
        # Extract hidden states from all transformer layers
        hidden_states = outputs.hidden_states  # Tuple of (layer_0, layer_1, ..., layer_23)
        
        # Convert tuple to list for easier manipulation
        # Stack hidden states along a new dimension (layer dimension)
        return torch.stack(hidden_states, dim=1)
    
if __name__ == "__main__":
    # Example usage
    model = WavLMWrapper()
    audio_signal = torch.randn(1, 16000*4)  # Example audio signal (batch_size=1, sequence_length=16000)
    hidden_states = model(audio_signal)
    
    # Print the shape of the hidden states from the last layer
    print(hidden_states.shape)  # Should be (1, 16000, 1024) for WavLM large
