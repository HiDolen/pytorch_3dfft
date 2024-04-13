import numpy as np
import torch


def to_numpy(array):
    """
    将任意类型的数组转换为 numpy 数组。
    """
    if torch.is_tensor(array):
        return array.cpu().numpy()
    return np.asarray(array)
