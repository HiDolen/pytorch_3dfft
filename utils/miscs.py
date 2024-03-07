import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt


def peak_grouping(detMat2D, plot=False):
    """
    通过使用 3x3 的卷积核，找出检测矩阵中的峰值点。

    输入维度：[fft_vel, fft_rang]
    
    输出维度：[2, points_count]
    """
    kernel = np.ones((3, 3))
    maxFilter = maximum_filter(detMat2D, footprint=kernel)
    peaks = np.where((detMat2D == maxFilter) & (detMat2D != 0))
    # 转换为 [points_count, 2] 堆叠的形式
    peaks = np.stack((peaks[0], peaks[1]), axis=0)

    # 绘制峰值点
    if plot:
        plt.figure()
        plt.imshow(detMat2D.T)
        plt.scatter(peaks[0, :], peaks[1, :], c='#ff3800', marker='x', s=20)
        plt.show()

    return peaks


