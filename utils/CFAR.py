import numpy as np
import math

from scipy.signal import convolve2d
import matplotlib.pyplot as plt


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


def cfar_os_2d(data, g, t, k, pfa=1e-5):
    """
    2D OS-CFAR detector
    输入数据维度：[fft_vel, fft_rang]

    Args:
        data: 2D data。应为实数数据
        g: 守护单元数量
        t: 训练单元数量
        k: 噪声单元的选择次序（升序排序）。一般是训练数量 n 的 3/4
        pfa: 虚警概率
        warp_boundary: 是否使用环绕边界
    """
    if np.iscomplexobj(data):
        raise ValueError("cfar 时 data 不允许为复数。")
    
    data_shape = data.shape

    # 将标量转换为元组
    g = np.tile(g, 2) if np.isscalar(g) else g
    t = np.tile(t, 2) if np.isscalar(t) else t

    window_size = (2 * g[0] + 2 * t[0] + 1, 2 * g[1] + 2 * t[1] + 1)
    window_sum = np.prod(window_size)
    g_sum = np.prod(2 * g + 1)

    alpha = os_cfar_threshold(k, window_sum - g_sum, pfa)

    window = np.ones(window_size, dtype=bool)
    window[
        t[0] : t[0] + 2 * g[0] + 1,
        t[1] : t[1] + 2 * g[1] + 1,
    ] = False

    # 对数据进行填充
    pad_width = ((g[0] + t[0], g[0] + t[0]), (g[1] + t[1], g[1] + t[1]))
    padded_data = np.pad(data, pad_width, mode='wrap')

    # TODO


def cfar_ca2d(data, g, n, pfa, *, warp_boundary=False, edge_crop=(0, 0), plot=False):
    """
    2D CA-CFAR detector
    输入数据维度：[fft_vel, fft_rang]

    Args:
        data: 2D data
        g: number of guard cells
        n: number of noise cells
        pfa: probability of false alarm
        warp_boundary: whether to use wrap boundary
        edge_crop: tuple of number of edge cells to crop in row and column
    """

    # 确保 g, n 和 edge_crop 是元组
    ensure_tuple = lambda x: x if isinstance(x, tuple) else (x, x)
    g, n, edge_crop = map(ensure_tuple, [g, n, edge_crop])

    # 创建卷积核
    kernel_size = (2 * (g[0] + n[0]) + 1, 2 * (g[1] + n[1]) + 1)
    kernel = np.ones(kernel_size)
    kernel[
        n[0] : n[0] + 2 * g[0] + 1,
        n[1] : n[1] + 2 * g[1] + 1,
    ] = 0

    # 计算噪声单元格的数量和阈值因子
    n = np.sum(kernel)
    alpha = n * (pfa ** (-1 / n) - 1)

    # 计算每个单元格的噪声水平
    if warp_boundary:
        noise_levels = convolve2d(data, kernel, mode='same', boundary='wrap') / n
    else:
        noise_levels = convolve2d(data, kernel, mode='same', boundary='fill')
        norm_matrix = convolve2d(np.ones(data.shape), kernel, mode='same', boundary='fill')
        noise_levels /= norm_matrix

    # 计算阈值并检测目标
    thresholds = alpha * noise_levels
    detections = data > thresholds

    # 去除边缘
    if edge_crop[0] > 0:
        detections[: edge_crop[0], :] = 0
        detections[-edge_crop[0] :, :] = 0
    if edge_crop[1] > 0:
        detections[:, : edge_crop[1]] = 0
        detections[:, -edge_crop[1] :] = 0

    if plot:
        # 绘制 data、thresholds 和 detections 图像
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(data.T)
        axs[0].set_title('Data')
        axs[1].imshow(thresholds.T)
        axs[1].set_title('Thresholds')
        axs[2].imshow(detections.T)
        axs[2].set_title('Detections')
        plt.show()

    cfar_result = np.where(detections == 1, data, 0)

    return cfar_result
