import torch
from torch import nn
import numpy as np
from einops import rearrange


class _FFTBase(nn.Module):
    """fft 操作。可自由选择 fft 维度"""

    def __init__(self, n, cube_dims, fft_dim, window=None):
        """
        args:
            n: fft 变换点数
            cube_dims: 输入数据的维度数量
            fft_dim: fft 施加维度
            window: 加窗序列
        """
        super().__init__()
        self.fft_dim = fft_dim
        self.n = n

        if window is None:
            window = torch.ones(1)
        elif isinstance(window, np.ndarray):
            window = torch.from_numpy(window)
        shape = [1] * cube_dims
        shape[fft_dim] = window.shape[0]
        self.window = window.reshape(shape)
        self.window = nn.Parameter(self.window, requires_grad=False)

    @torch.no_grad()
    def forward(self, X):
        X *= self.window
        return torch.fft.fft(X, self.n, dim=self.fft_dim)


class RangeFFT(_FFTBase):
    """range fft 操作。快时间维度处理。

    默认输入维度：[Rx, chirp_loop_count, fft_rang]

    默认 fft 维度：2
    """

    def __init__(self, n, cube_dims=3, fft_dim=2, window=None):
        super().__init__(n, cube_dims, fft_dim, window)


class DopplerFFT(_FFTBase):
    """doppler fft 操作。慢时间维度处理。

    默认输入维度：[Rx, chirp_loop_count, fft_rang]

    默认 fft 维度：1
    """

    def __init__(self, n, cube_dims=3, fft_dim=1, window=None):
        super().__init__(n, cube_dims, fft_dim, window)

    @torch.no_grad()
    def forward(self, X):
        output = super().forward(X)
        return torch.fft.fftshift(output, dim=self.fft_dim)


class AngleFFT(_FFTBase):
    """angle fft 操作。角度维度处理。

    默认输入维度：[Rx, chirp_loop_count, fft_rang]

    默认 fft 维度：0
    """

    def __init__(self, n, cube_dims=3, fft_dim=0, window=None):
        super().__init__(n, cube_dims, fft_dim, window)

    @torch.no_grad()
    def forward(self, X):
        output = super().forward(X)
        return torch.fft.fftshift(output, dim=self.fft_dim)


class PhaseCompensation(nn.Module):
    """相位补偿操作"""

    def __init__(self, dop_bin, N_tx):
        """补偿多 Tx 发射的相位差

        Args:
            dop_bin: doppler fft 点数
            N_tx: 发射天线数量
        """
        super().__init__()
        self.N_tx = N_tx

        dop_bins = torch.arange(dop_bin)

        delta_phi = -2j * torch.pi * (dop_bins - dop_bin / 2) / dop_bin / N_tx
        delta_phi = delta_phi.reshape(1, -1, 1) # 对 fft_vel 进行补偿
        delta_phi = torch.stack([delta_phi * i for i in range(N_tx)], dim=0)
        delta_phi = torch.exp(delta_phi)
        self.delta_phi = nn.Parameter(delta_phi, requires_grad=False)

    @torch.no_grad()
    def forward(self, X):
        """
        输入数据维度：[batch, Tx, Rx, fft_vel, fft_rang]，或者 [Tx, Rx, fft_vel, fft_rang]
        """
        return X * self.delta_phi
