import torch


class SeparableConv2d(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, onnx_compatible=False):
        """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
        """
        super(SeparableConv2d, self).__init__(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, groups=in_channels,
                      stride=stride, padding=padding),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1))
