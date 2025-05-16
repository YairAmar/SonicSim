import torch

class MagPhaSTFT(torch.nn.Module):
    def __init__(self, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
        super(MagPhaSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.center = center
        self.hann_window = torch.hann_window(win_size)

    def forward(self, y):
        stft_spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                               center=self.center, pad_mode='reflect', normalized=False, return_complex=True)
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
        # Magnitude Compression
        mag = torch.pow(mag, self.compress_factor)
        com = mag * torch.exp(1j * pha)

        return mag, pha, com


class MagPhaISTFT(torch.nn.Module):
    def __init__(self, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
        super(MagPhaISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.center = center
        self.hann_window = torch.hann_window(win_size)

    def forward(self, mag, pha):
        # Magnitude Decompression
        mag = torch.pow(mag, (1.0 / self.compress_factor))
        com = mag * torch.exp(1j * pha)
        wav = torch.istft(com, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.hann_window, center=self.center)

        return wav
