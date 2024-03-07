import torch


def complex_abs_sub_2d(c, real):
    """复数的幅度减法，针对二维数据

    复数其幅度为原来的幅度减去这个实数。

    Args:
        c: 维度 [batch, ..., x, y]
        real: 维度 [batch, x, y]
    """
    magnitude = torch.abs(c)
    vec = c / (magnitude + 1e-8) # 获得单位向量

    real_shape = list(real.shape)
    for _ in range(len(c.shape) - len(real.shape)):
        real_shape.insert(1, 1)
    real = real.view(real_shape)

    new_magnitude = magnitude - real # 在这个方向上减去实数
    new_magnitude = torch.clamp(new_magnitude, min=0)  # 不小于0
    new_c = new_magnitude * vec

    return new_c
