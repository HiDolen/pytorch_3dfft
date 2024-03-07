import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.constants import c as speed_of_light


def get_range_grid(sample_count, sampling_rate, slope):
    """
    生成 range-fft 对应的坐标轴

    args:
        sample_count: 采样点的数量
        sampling_rate: 采样率
        slope: 扫频斜率
    return:
        range_grid:
    """
    resolution = sampling_rate / sample_count  # 采样点的频率间隔
    freq_grid = np.arange(sample_count) * resolution  # 频率网格
    # 用公式 R = cTf / 2B，转换为 range-fft 的坐标轴
    range_grid = freq_grid * c / slope / 2
    return range_grid


def get_velocity_grid(sample_count, sweep_time, fc):
    """
    生成 velocity-fft 对应的坐标轴

    args:
        sample_count: 采样点的数量
        sweep_time: 扫频时间。作为采样率
        fc: chirp 信号的起始频率
    """
    sample_interval = 1 / sweep_time  # 采样点的时间间隔
    dop_grid = np.fft.fftshift(np.fft.fftfreq(sample_count, sample_interval))
    lambda_ = c / fc  # 波长
    # 用公式 v = dop * lambda / 2，转换为 velocity-fft 的坐标轴
    vel_grid = dop_grid * lambda_ / 2
    return vel_grid


def plot_rangeDoppler(data, rng_grid, vel_grid):
    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0)

    vel_grid, rng_grid = np.meshgrid(vel_grid, rng_grid)

    ax.plot_surface(vel_grid, rng_grid, data, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Doppler Velocity (m/s)')
    ax.set_ylabel('Range(meters)')
    ax.set_zlim(0, 3e04)
    ax.set_title('Range-Doppler heatmap')
    plt.show()


class RadarPlot:
    def __init__(self, fc, S, Tc, Fs, fft_rang, fft_vel, fft_ang):
        """
        用于辅助绘制雷达数据

        args:
            fc: chirp 信号的起始频率
            S: 扫频斜率
            Tc: 一个 chirp 信号的时间
            Fs: 采样率
            fft_rang: range-fft 的点数
            fft_vel: velocity-fft 的点数
            fft_ang: angle-fft 的点数
        """
        self.fc = fc
        self.S = S
        self.Tc = Tc
        self.Fs = Fs
        self.fft_rang = fft_rang
        self.fft_vel = fft_vel
        self.fft_ang = fft_ang

        self.rng_grid = self._get_rng_grid()
        self.vel_grid = self._get_vel_grid()
        self.ang_grid = self._get_ang_grid()

    def plot_rangeDoppler(self, data):
        """
        输入 range-Doppler 数据，绘制 range-Doppler 热力图
        输入维度：[fft_vel, fft_rang]
        """
        # 将复数转换为幅度
        data = np.abs(data)

        fig, ax = plt.subplots(figsize=(7, 5.6))

        rng_grid, vel_grid = np.meshgrid(self.rng_grid, self.vel_grid)

        c = ax.pcolormesh(vel_grid, rng_grid, data, cmap='viridis', vmax=3e04)
        fig.colorbar(c, ax=ax)

        ax.set_xlabel('Doppler Velocity (m/s)')
        ax.set_ylabel('Range(meters)')
        ax.set_title('Range-Doppler heatmap')
        plt.show()

    def plot_3DFFT(self, fft_rng, fft_vel, fft_ang):
        """
        绘制 3D-FFT 的点云图
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 使用 grid 对数据进行映射
        vel = self.vel_grid[fft_vel]
        ang = self.ang_grid[fft_ang]
        rng = self.rng_grid[fft_rng]
        ax.scatter3D(vel, ang, rng)
        ax.set_xlabel('Doppler velocity (m/s)')
        ax.set_ylabel('Azimuth angle (degrees)')
        ax.set_zlabel('Range (m)')
        # 设置坐标轴的范围
        ax.set_xlim([-5, 5])
        ax.set_ylim([-60, 60])
        ax.set_zlim([0, 30])
        ax.set_title('3D point clouds')
        plt.grid(True)
        plt.show()

    def _get_rng_grid(self):
        """
        生成 range-fft 对应的坐标轴
        """
        freq_res = self.Fs / self.fft_rang  # 每个采样点的频率间隔
        freq_grid = np.arange(self.fft_rang) * freq_res  # 频率网格
        # 用公式 R = cTf / 2B，转换为 range-fft 的坐标轴
        range_grid = freq_grid * speed_of_light / self.S / 2
        return range_grid

    def _get_vel_grid(self):
        """
        生成 velocity-fft 对应的坐标轴
        """
        sample_interval = self.Tc  # 慢时间维度的采样点的时间间隔
        dop_grid = np.fft.fftshift(np.fft.fftfreq(self.fft_vel, sample_interval))
        # 用公式 v = dop * lambda / 2，转换为 velocity-fft 的坐标轴
        lambda_ = speed_of_light / self.fc
        vel_grid = dop_grid * lambda_ / 2
        return vel_grid

    def _get_ang_grid(self):
        """
        生成 angle-fft 对应的坐标轴
        """
        w = np.linspace(-1, 1, self.fft_ang)
        ang_grid = np.arcsin(w) * 180 / np.pi
        return ang_grid
