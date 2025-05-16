import torch
from look2hear.utils.mpsenet_stft import MagPhaISTFT, MagPhaSTFT
from look2hear.layers.mpsenet_tranformer import DenseEncoder, TSTransformerBlock, MaskDecoder, PhaseDecoder


class MPSENet(torch.nn.Module):
    def __init__(self, 
                 dense_channel, 
                 n_fft, 
                 beta, 
                 compress_factor,
                 hop_size,
                 win_size, 
                 num_tsblocks=4, 
                 n_heads=4):
        super(MPSENet, self).__init__()
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(dense_channel=dense_channel, in_channel=2)

        self.TSTransformer = torch.nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(dense_channel=dense_channel, n_heads=n_heads))
        
        self.mask_decoder = MaskDecoder(dense_channel=dense_channel, n_fft=n_fft, beta=beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel=dense_channel, out_channel=1)
        self.stft = MagPhaSTFT(n_fft=n_fft, hop_size=hop_size, win_size=win_size, compress_factor=compress_factor)
        self.istft = MagPhaISTFT(n_fft=n_fft, hop_size=hop_size, win_size=win_size, compress_factor=compress_factor)

    def forward(self, x): # [B, F, T]
        norm_factor = torch.sqrt(len(x) / torch.sum(x ** 2.0))
        x = (x * norm_factor)
        noisy_amp, noisy_pha, _ = self.stft(x)
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1) # [B, 2, T, F]
        
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)
        
        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack((denoised_amp*torch.cos(denoised_pha),
                                    denoised_amp*torch.sin(denoised_pha)), dim=-1)
        
        audio_g = self.istft(denoised_amp, denoised_pha)
        audio_g = audio_g / norm_factor

        return denoised_amp, denoised_pha, denoised_com, audio_g


# if __name__ == "__main__":
#     import torch
#     from look2hear.utils.mpsenet_stft import MagPhaISTFT, MagPhaSTFT
#     from look2hear.layers.mpsenet_tranformer import DenseEncoder, TSTransformerBlock, MaskDecoder, PhaseDecoder
#     from look2hear.configs.mpsenet_config import hparams as h

#     model = MPNet(h)
#     x = torch.randn(1, 400, 100)
#     denoised_amp, denoised_pha, denoised_com, audio_g = model(x)
#     print(denoised_amp.shape, denoised_pha.shape, denoised_com.shape, audio_g.shape)
