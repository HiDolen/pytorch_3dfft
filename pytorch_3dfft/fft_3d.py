import numpy as np
import plotly.graph_objects as go
import torch
from einops.layers.torch import Rearrange
from scipy import signal
from scipy.constants import c as speed_of_light
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


class Plot:
    """进行 3D-FFT 后，可用该函数绘制热力图。

    与 plot() 函数不同，这个类包含雷达特征，可以绘制具体数值的热力图。
    """

    def __init__(
        self,
        center_freq,
        bandwidth,
        pulse_duration,
        pulse_period,
        sampling_freq,
        fft_rng,
        fft_vel,
        fft_ang,
    ):
        """
        Args:
            center_freq: $f_c$，信号中心频率
            bandwidth: $B$，发射的调频信号的带宽
            pulse_duration: $T_c$，一次脉冲的持续时间
            pulse_period: 脉冲重复周期。每隔这段时间发射一次脉冲
            sampling_freq: $F_s$，采样频率
            fft_rng: range fft 采样点数
            fft_vel: velocity fft 采样点数
            fft_ang: angle fft 采样点数
        """
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.pulse_duration = pulse_duration
        self.pulse_period = pulse_period
        self.sampling_freq = sampling_freq
        self.fft_rng = fft_rng
        self.fft_vel = fft_vel
        self.fft_ang = fft_ang

        self.range_grid = self._get_rng_grid()
        self.vel_grid = self._get_vel_grid()
        self.ang_grid = self._get_ang_grid()

    def plot(self, x: torch.Tensor, mean_dim=0, *, z_bounds=None, height=400, width=700):
        """绘制热力图

        Args:
            x: (tx * rx, chirps, samples)，不区分是复数还是实数
            mean_dim: 计算均值的维度

            z_bounds: 热力图的 z 轴范围。默认为 [-10, 25]
            height: 热力图高度
            width: 热力图宽度
        """
        assert mean_dim in [0, 1, 2]
        assert len(x.shape) == 3

        if z_bounds is None:
            z_bounds = [-10, 25]

        x = x.cpu().numpy()
        z = 20 * np.log10(np.mean(np.abs(x + 1e-8), axis=mean_dim))
        if mean_dim == 0:
            title = "Range-Doppler Heatmap"
            xaxis = dict(title="Doppler")
            yaxis = dict(title="Range")
            x = self.vel_grid
            y = self.range_grid
            z = z.T
        elif mean_dim == 1:
            title = "Range-Angle Heatmap"
            xaxis = dict(title="Angle")
            yaxis = dict(title="Range")
            x = self.ang_grid
            y = self.range_grid
            z = z.T
        else:
            title = "Doppler-Angle Heatmap"
            xaxis = dict(title="Angle")
            yaxis = dict(title="Doppler")
            x = self.ang_grid
            y = self.vel_grid
            z = z.T
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                colorbar={"title": "dB"},
                zmin=z_bounds[0],
                zmax=z_bounds[1],
            )
        )
        fig.update_layout(
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            height=height,
            width=width,
        )
        fig.show()

    def _get_rng_grid(self):
        """
        生成 range-fft 对应的坐标轴
        """
        # # 用公式 R = cTf / 2B，转换为 range-fft 的坐标轴
        max_range = speed_of_light * self.sampling_freq * self.pulse_duration / self.bandwidth / 2
        range_grid = np.linspace(0, max_range, self.fft_rng, endpoint=False)
        return range_grid

    def _get_vel_grid(self):
        """
        生成 velocity-fft 对应的坐标轴
        """
        # # 用公式 v = dop * lambda / 2，转换为 velocity-fft 的坐标轴
        dop_grid = np.linspace(
            -1 / self.pulse_duration / 2, 1 / self.pulse_duration / 2, self.fft_vel, endpoint=False
        )
        vel_grid = dop_grid * speed_of_light / self.center_freq / 2
        return vel_grid

    def _get_ang_grid(self):
        """
        生成 angle-fft 对应的坐标轴
        """
        w = np.linspace(-1, 1, self.fft_ang)
        ang_grid = np.arcsin(w) * 180 / np.pi
        return ang_grid


def plot(x, mean_dim=0, *, z_bounds=None, height=400, width=700):
    """进行 3D-FFT 后，可用该函数绘制热力图

    Args:
        x: (tx * rx, chirps, samples)，不区分是复数还是实数
        mean_dim: 计算均值的维度

        z_bounds: 热力图的 z 轴范围
        height: 热力图高度
        width: 热力图宽度
    """
    assert mean_dim in [0, 1, 2]
    assert len(x.shape) == 3

    if z_bounds is None:
        z_bounds = [None, None]

    z = 20 * np.log10(torch.mean(torch.abs(x + 1e-8), dim=mean_dim).cpu().numpy())
    if mean_dim == 0:
        title = "Range-Doppler Heatmap"
        xaxis = dict(title="Doppler")
        yaxis = dict(title="Range")
        z = z.T
    elif mean_dim == 1:
        title = "Range-Angle Heatmap"
        xaxis = dict(title="Angle")
        yaxis = dict(title="Range")
        z = z.T
    else:
        title = "Doppler-Angle Heatmap"
        xaxis = dict(title="Angle")
        yaxis = dict(title="Doppler")
        z = z.T
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale="Viridis",
            colorbar={"title": "dB"},
            zmin=z_bounds[0],
            zmax=z_bounds[1],
        )
    )
    fig.update_layout(
        title=title,
        xaxis=xaxis,
        yaxis=yaxis,
        height=height,
        width=width,
    )
    fig.show()
