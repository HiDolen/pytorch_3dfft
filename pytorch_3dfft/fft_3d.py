import torch
from einops.layers.torch import Rearrange
from scipy import signal
from torch import nn

from .utils.ffts import AngleFFT, DopplerFFT, PhaseCompensation, RangeFFT


class RadarData(nn.Module):
    def __init__(self, raw_shape, fft_shape, cfar_module=None):
        """雷达数据 3D-FFT 处理。

        Args:
            raw_shape: (tx, rx, chirps, samples)
            fft_shape: (angle, doppler, range)
            cfar_module: 用 nn.Module 包装的 CFAR 模块
        """
        super(RadarData, self).__init__()

        tx, rx, chirps, samples = raw_shape
        angle_bin, doppler_bin, range_bin = fft_shape
        self.raw_shape = raw_shape
        self.fft_shape = fft_shape

        range_window = signal.windows.chebwin(samples, at=80)
        doppler_window = signal.windows.chebwin(chirps, at=60)
        angle_window = signal.windows.chebwin(tx * rx, at=50)

        layers = [
            RangeFFT(range_bin, 5, 4, range_window),
            DopplerFFT(doppler_bin, 5, 3, doppler_window),
            PhaseCompensation(doppler_bin, tx),
            Rearrange("b t r c s -> b (t r) c s"),
            AngleFFT(angle_bin, 4, 1, angle_window),
        ]
        if cfar_module is not None:
            layers.insert(2, cfar_module)

        self.fft_3d = nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, x):
        """

        Args:
            x: (batch, tx, rx, doppler, range)
        """
        return self.fft_3d(x.conj())



