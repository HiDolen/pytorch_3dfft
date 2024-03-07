# %%
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class _CFARBase_2D(nn.Module):
    def __init__(self, g, t, pfa=1e-5):
        """
        2D CFAR 基类

        arg:
            g: 守护单元数量
            t: 训练单元数量
            pfa: 虚警概率

        g 和 t 若是二维元组，会分别对应倒数第 2 维和倒数第 1 维（一般是 fft_vel 和 fft_rang）。
        """
        super(_CFARBase_2D, self).__init__()
        self.g = np.tile(g, 2) if np.isscalar(g) else g
        self.t = np.tile(t, 2) if np.isscalar(t) else t
        self.pfa = pfa

        self.kernel_size = (
            2 * (self.g[0] + self.t[0]) + 1,
            2 * (self.g[1] + self.t[1]) + 1,
        )
        self.kernel = torch.ones(self.kernel_size, dtype=bool)
        self.kernel[
            self.t[0] : self.t[0] + 2 * self.g[0] + 1,
            self.t[1] : self.t[1] + 2 * self.g[1] + 1,
        ] = False
        self.n = self.kernel.sum().item()


class CFAR_OSCA_2D_old(_CFARBase_2D):
    """
    针对 range-doppler 的二维 CFAR。
    """
    def __init__(self, g, t, pfa=None, k=None):
        """针对 range-doppler 的二维 CFAR。
        
        输入数据维度：[batch, fft_vel, fft_rang]

        Args:
            g: 守护单元数量
            t: 训练单元数量
            pfa: 虚警概率
            k: 噪声单元的选择次序（升序排序）。一般是训练数量 n 的 3/4
        """
        super(CFAR_OSCA_2D, self).__init__(g, t, pfa)

        self.os_kernel = torch.ones(self.kernel_size[1], dtype=bool)
        self.os_kernel[self.t[1] : self.t[1] + 2 * self.g[1] + 1] = False
        self.ca_kernel = torch.ones(self.kernel_size[0], dtype=bool)
        self.ca_kernel[self.t[0] : self.t[0] + 2 * self.g[0] + 1] = False
        self.ca_kernel = rearrange(self.ca_kernel, "v -> 1 1 v 1")

        self.os_n = self.os_kernel.sum().item()
        self.ca_n = self.ca_kernel.sum().item()

        self.k = self.os_n * 3 / 4 if k is None else k

        self.os_alpha = os_cfar_threshold(self.k, self.os_n, self.pfa)
        self.os_alpha = np.sqrt(self.os_alpha)

        # os_cfar 专用
        pad_width = (self.g[1] + self.t[1], self.g[1] + self.t[1], 0, 0)
        self.layer_os_pad = nn.CircularPad2d(pad_width)
        unfold_kernel_size = (1, self.os_kernel.shape[0])
        self.layer_os_unfold = nn.Unfold(kernel_size=unfold_kernel_size, stride=1)

        # ca_cfar 专用
        pad_width = (0, 0, self.g[0] + self.t[0], self.g[0] + self.t[0])
        self.layer_ca_pad = nn.CircularPad2d(pad_width)

        self.rearrange_bvr_to_b1vr = Rearrange("b v r -> b 1 v r")
        self.rearrange_b1vr_to_bvr = Rearrange("b 1 v r -> b v r")

    @torch.no_grad()
    def forward(self, data, dim=-1):
        """
        输入数据维度：[batch, fft_vel, fft_rang]
        """
        batch_size, fft_vel, fft_rang = data.size()

        data = self.layer_os_pad(data)
        # data = rearrange(data, "b v r -> b 1 v r")
        data = self.rearrange_bvr_to_b1vr(data)
        windows = self.layer_os_unfold(data)
        windows = windows[:, self.os_kernel, :]

        miu = windows.topk(int(self.os_n - self.k), dim=1).values[:, -1]
        miu = miu.view(batch_size, fft_vel, fft_rang)
        os_result = self.os_alpha * miu

        # 再进行 CA
        padded = self.layer_ca_pad(os_result)
        # padded = rearrange(padded, "b v r -> b 1 v r")
        padded = self.rearrange_bvr_to_b1vr(padded)
        ca_result = F.conv2d(padded, self.ca_kernel.type_as(data)) / self.ca_n
        # ca_result = rearrange(ca_result, "b 1 v r -> b v r")
        ca_result = self.rearrange_b1vr_to_bvr(ca_result)

        return ca_result


class CFAR_OSCA_2D_new(_CFARBase_2D):
    """
    针对 range-doppler 的二维 CFAR。
    """

    def __init__(self, g, t, pfa=None, k=None):
        """针对 range-doppler 的二维 CFAR。

        输入数据维度：[batch, tx, rx, fft_vel, fft_rang]

        Args:
            g: 守护单元数量
            t: 训练单元数量
            pfa: 虚警概率
            k: 噪声单元的选择次序（升序排序）。一般是训练数量 n 的 3/4
        """
        super().__init__(g, t, pfa)

        self.os_kernel = torch.ones(self.kernel_size[1], dtype=bool)
        self.os_kernel[self.t[1] : self.t[1] + 2 * self.g[1] + 1] = False
        self.ca_kernel = torch.ones(self.kernel_size[0], dtype=bool)
        self.ca_kernel[self.t[0] : self.t[0] + 2 * self.g[0] + 1] = False
        self.ca_kernel = rearrange(self.ca_kernel, "v -> 1 1 v 1")

        self.os_n = self.os_kernel.sum().item()
        self.ca_n = self.ca_kernel.sum().item()

        self.k = self.os_n * 3 / 4 if k is None else k

        self.os_alpha = os_cfar_threshold(self.k, self.os_n, self.pfa)
        self.os_alpha = np.sqrt(self.os_alpha)

        # os_cfar 专用
        pad_width = (self.g[1] + self.t[1], self.g[1] + self.t[1], 0, 0)
        self.layer_os_pad = nn.CircularPad2d(pad_width)
        unfold_kernel_size = (1, self.os_kernel.shape[0])
        self.layer_os_unfold = nn.Unfold(kernel_size=unfold_kernel_size, stride=1)

        # ca_cfar 专用
        pad_width = (0, 0, self.g[0] + self.t[0], self.g[0] + self.t[0])
        self.layer_ca_pad = nn.CircularPad2d(pad_width)

        self.complex_abs_sub_2d = ComplexAbsSub2D(5)

        self.rearrange_bvr_to_b1vr = Rearrange("b v r -> b 1 v r")
        self.rearrange_b1vr_to_bvr = Rearrange("b 1 v r -> b v r")

    @torch.no_grad()
    def forward(self, x):
        """
        输入数据维度：[batch, tx, rx, fft_vel, fft_rang]
        """
        batch_size, tx, rx, fft_vel, fft_rang = x.size()
        data = torch.mean(torch.abs(x.view(batch_size, -1, fft_vel, fft_rang)), axis=1)

        # pad 与 unfold
        data = self.layer_os_pad(data)
        data = self.rearrange_bvr_to_b1vr(data)
        windows = self.layer_os_unfold(data)
        windows = windows[:, self.os_kernel, :]

        # 先进行 OS
        miu = windows.topk(int(self.os_n - self.k), dim=1).values[:, -1]
        miu = miu.view(batch_size, fft_vel, fft_rang)
        os_result = self.os_alpha * miu

        # 再进行 CA
        padded = self.layer_ca_pad(os_result)
        padded = self.rearrange_bvr_to_b1vr(padded)
        ca_result = F.conv2d(padded, self.ca_kernel.type_as(data)) / self.ca_n
        ca_result = self.rearrange_b1vr_to_bvr(ca_result)

        return self.complex_abs_sub_2d(x, ca_result)



class CFAR_OS_2D(_CFARBase_2D):
    def __init__(self, g, t, pfa=None, k=None):
        """
        2D OS-CFAR detector
        输入数据维度：[fft_vel, fft_rang]

        Args:
            data: 2D data。应为实数数据
            g: 守护单元数量
            t: 训练单元数量
            pfa: 虚警概率
            k: 噪声单元的选择次序（升序排序）。一般是训练数量 n 的 3/4
        """
        super(CFAR_OS_2D, self).__init__(g, t, pfa)

        self.k = self.n * 3 / 4 if k is None else k
        self.alpha = os_cfar_threshold(self.k, self.n, self.pfa)

        self.pad_width = (
            self.g[0] + self.t[0],
            self.g[0] + self.t[0],
            self.g[1] + self.t[1],
            self.g[1] + self.t[1],
        )
        self.layer_pad = nn.CircularPad2d(self.pad_width)

    @torch.no_grad()
    def forward(self, data):
        """
        输入数据维度：[batch, fft_vel, fft_rang]
        """
        shape = data.shape

        data = self.layer_pad(data)
        cfar_result = torch.zeros(shape[1:])
        for i in range(self.pad_width[0], shape[1] + self.pad_width[1]):
            for j in range(self.pad_width[2], shape[2] + self.pad_width[3]):
                window = data[
                    :,
                    i - self.pad_width[0] : i + self.pad_width[1] + 1,
                    j - self.pad_width[2] : j + self.pad_width[3] + 1,
                ]
                window = window[:, self.kernel]
                miu = window.topk(self.n - self.k, dim=1).values[:, -1]
                cfar_result[i - self.pad_width[0], j - self.pad_width[2]] = miu

        return self.alpha * cfar_result


def os_cfar_threshold(k, n, pfa):
    """计算 alpha"""

    def log_factorial(n):
        """用于近似计算阶乘的对数。

        计算 0, 1, 2, ..., n-1 的阶乘的对数。
        """
        n = n + 1
        if n < 9:
            return np.log(math.factorial(n))
        return 1 / 2 * (np.log(2 * np.pi) - np.log(n)) + n * (
            np.log(n + 1 / (12 * n - (1 / 10 / n))) - 1
        )

    def fun(k, n, t_os, pfa):
        return (
            log_factorial(n)
            - log_factorial(n - k)
            - np.sum(np.log(np.arange(n, n - k, -1) + t_os))
            - np.log(pfa)
        )

    max_iter = 10000
    t_max = 1e32
    t_min = 1
    for _ in range(0, max_iter):
        m_n = t_max - fun(k, n, t_max, pfa) * (t_min - t_max) / (
            fun(k, n, t_min, pfa) - fun(k, n, t_max, pfa)
        )
        f_m_n = fun(k, n, m_n, pfa)
        if f_m_n == 0 or np.abs(t_max - t_min) < 0.0001:
            return m_n

        if fun(k, n, t_max, pfa) * f_m_n < 0:
            t_min = m_n
        elif fun(k, n, t_min, pfa) * f_m_n < 0:
            t_max = m_n
        else:
            break

    raise ValueError("CFAR阈值计算不收敛。")

class ComplexAbsSub2D(nn.Module):
    """复数的幅度减法，针对二维数据

    复数其幅度为原来的幅度减去这个实数。

    Args:
        c: 维度 [batch, ..., x, y]
        real: 维度 [batch, x, y]
    """
    def __init__(self, input_dim_num):
        """
        
        Args:
            input_dim_num: 输入数据的维度总数。例如输入数据维度为 [batch, tx, rx, x, y]，则为 5
        """
        super().__init__()
        self.input_dim_num = input_dim_num

    @torch.no_grad()
    def forward(self, c, real):
        """
        Args:
            c: 维度 [batch, ..., x, y]
            real: 维度 [batch, x, y]
        """
        magnitude = torch.abs(c)
        vec = c / (magnitude + 1e-8)  # 获得单位向量

        for _ in range(self.input_dim_num - 3):
            real.unsqueeze_(1)

        new_magnitude = magnitude - real  # 在这个方向上减去实数
        new_magnitude = torch.clamp(new_magnitude, min=0)  # 不小于0
        new_c = new_magnitude * vec

        return new_c


def complex_abs_sub_2d(c, real):
    """复数的幅度减法，针对二维数据

    复数其幅度为原来的幅度减去这个实数。

    Args:
        c: 维度 [batch, ..., x, y]
        real: 维度 [batch, x, y]
    """
    magnitude = torch.abs(c)
    vec = c / (magnitude + 1e-8)  # 获得单位向量

    real_shape = list(real.shape)
    for _ in range(len(c.shape) - len(real.shape)):
        real_shape.insert(1, 1)
    real = real.view(real_shape)

    new_magnitude = magnitude - real  # 在这个方向上减去实数
    new_magnitude = torch.clamp(new_magnitude, min=0)  # 不小于0
    new_c = new_magnitude * vec

    return new_c
