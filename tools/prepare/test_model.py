import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        squeeze_channels (None or int): The intermediate channel number of
            SElayer. Default: None, means the value of ``squeeze_channels``
            is ``make_divisible(channels // ratio, divisor)``.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will
            be ``make_divisible(channels // ratio, divisor)``. Only used when
            ``squeeze_channels`` is None. Default: 16.
        divisor(int): The divisor to true divide the channel number. Only
            used when ``squeeze_channels`` is None. Default: 8.
        conv_cfg (None or dict): Config dict for convolution layer. Default:
            None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 squeeze_channels=24,
                 bias='auto',
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        print(out.shape)
        out = self.conv1(out)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        return x * out

if __name__ == '__main__':
    # global_avgpool = nn.AdaptiveAvgPool2d(1)
    # x = torch.randn((16, 197, 197, 12))
    # x = x.permute(0, 3, 1, 2).contiguous()
    # out = global_avgpool(x)
    # print(x.shape)
    # print(out.shape)
    # arr = []
    # for i in range(10):
    #     arr.append(torch.randn((2, 3, 4)))
    # out = torch.stack(arr, dim=0)
    # print(out.shape)
    #[num_layer, B, num_heads, N, N]
    K, B, C, N, N = 32, 197, 12, 16, 16
    x = torch.randn((B, C, N, N))
    x = torch.sum(x, dim=1)
    print(x.shape)

