import numpy as np


def range_fft(X, n=None, window=None):
    """
    将原始数据进行 fft 变换。快时间维度处理
    输入 X 的维度：[..., samples_count]
    args:
        X: 原始数据
        n: range fft 的点数
        window: 对输入数据加窗的窗口序列，如果为 None，则不加窗
    """
    if window is not None:
        shape = [1] * X.ndim
        shape[-1] = X.shape[-1]
        X = X * window.reshape(shape)
    output = np.fft.fft(X, n, axis=-1)
    return output


def doppler_fft(X, n=None, window=None):
    """
    将 fft_range 的结果进行 fft 变换。慢时间维度处理

    输入维度：[..., chirp_loop_count, fft_rang]
    """
    if window is not None:
        shape = [1] * X.ndim
        shape[-2] = X.shape[-2]
        X = X * window.reshape(shape)
    output = np.fft.fftshift(np.fft.fft(X, n, axis=-2), axes=-2)
    return output


def angle_fft(X, n=None, window=None):
    """
    将 fft_doppler 的结果进行 fft 变换。角度维度处理

    输入维度：[Rx, ...]
    """
    if window is not None:
        shape = [1] * X.ndim
        shape[0] = X.shape[0]
        X = X * window.reshape(shape)
    output = np.fft.fftshift(np.fft.fft(X, n, axis=0), axes=0)
    return output
