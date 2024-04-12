from setuptools import setup, find_packages

setup(
    name="pytorch_3dfft",
    version="0.1.1",
    description="3D FFT for PyTorch. Network layers inheritable from nn.Module.",
    packages=["pytorch_3dfft"],
    install_requires=[
        "numpy",
        "torch",
        "plotly",
        "scipy",
    ],
    author="HiDolen",
    author_email="820859278@qq.com"
)
