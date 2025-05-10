import torch


class UpstreamEncoder(torch.nn.Module):
    def __init__(self, upstream_model, model_path=None):
        super().__init__()
        if upstream_model == "wavlm":
            from look2hear.upstream_encoders.wavlm import WavLMWrapper
            self.upstream_model = WavLMWrapper(model_path=model_path)
        elif upstream_model == "wav2vec2":
            raise NotImplementedError("Wav2Vec2 is not implemented yet.")
            # from enhancement.look2hear.upstream_encoders.wav2vec2 import Wav2Vec2Wrapper
            # self.upstream_model = Wav2Vec2Wrapper()
        elif upstream_model == "hubert":
            # raise NotImplementedError("HuBERT is not implemented yet.")
            from look2hear.upstream_encoders.hubert import HuBERTWrapper
            self.upstream_model = HuBERTWrapper()

    def forward(self, x):
        """
        Encode the audio input.
        """
        # x: (batch_size, sequence_length)
        # upstream_x: (batch_size, feature_dim, sequence_length)
        upstream_x = self.upstream_model(x)
        return upstream_x
