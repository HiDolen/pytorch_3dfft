import numpy as np
from scipy.constants import c as speed_of_light


class RadarProcessor:
    def __init__(self, fc, S, Tc, fft_rang, fft_vel, fft_ang):
        """
        用于辅助处理雷达数据

        args:
            fc: chirp 信号的起始频率
            S: 扫频斜率
            Tc: 一个 chirp 信号的时间
            fft_rang: range-fft 的点数
            fft_vel: velocity-fft 的点数
            fft_ang: angle-fft 的点数
        """
        self.fc = fc
        self.S = S
        self.Tc = Tc
        self.fft_rang = fft_rang
        self.fft_vel = fft_vel
        self.fft_ang = fft_ang

        self.lambda_ = speed_of_light / fc  # 波长
        self.freq_res = 1 / Tc / fft_vel  # 每个采样点的频率间隔

        self.range_grid = self._get_range_grid()
        self.velocity_grid = self._get_velocity_grid()
        self.angle_grid = self._get_angle_grid()

    def _get_range_grid(self):
        """
        生成 range-fft 对应的坐标轴
        """
        freq_grid = np.arange(self.fft_rang) * self.freq_res  # 频率网格
        # 用公式 R = cTf / 2B，转换为 range-fft 的坐标轴
        range_grid = freq_grid * speed_of_light / self.S / 2
        return range_grid
    
    def _get_velocity_grid(self):
        """
        生成 velocity-fft 对应的坐标轴
        """
        sample_interval = 1 / self.Tc # 慢时间维度的采样点的时间间隔
        dop_grid = np.fft.fftshift(np.fft.fftfreq(self.fft_vel, sample_interval))
        # 用公式 v = dop * lambda / 2，转换为 velocity-fft 的坐标轴
        vel_grid = dop_grid * self.lambda_ / 2
        return vel_grid
    
    def _get_angle_grid(self):
        pass