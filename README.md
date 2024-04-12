# PyTorch 3DFFT

针对 TDM MIMO FMCW 雷达，3D-FFT 的 PyTorch 实现。可作为网络层放入到任意网络中。

## 处理过程

RangeFFT -> DopplerFFT -> cfar（可选） -> PhaseCompensation -> AngleFFT

输入维度应为 `[batch, tx, rx, chirp, sample]`。即，按照 batch、天线、慢时间维度、快时间维度进行排序。

输出维度为 `[batch, angle, doppler, range]`。

## 使用方法

### 安装

将本项目 clone 到本地，使用 `pip install -e .` 安装。

### 代码示例

准备好雷达输入，维度应为 `[batch, tx, rx, chirp, sample]`。

实例化网络：

```python
from pytorch_3dfft import RadarData
from pytorch_3dfft.utils.CFAR import CFAR_OSCA_2D

cfar = CFAR_OSCA_2D(3, (40, 20), 1e-9)
pre_process = RadarData((2, 4, 255, 128), (128, 128, 128), cfar_module=cfar)
```

其中，`cfar` 是继承于 `nn.Module` 用于 CFAR 处理的网络层。`(2, 4, 255, 128)` 指定了输入维度，`(64, 256, 128)` 指定了输出维度。
