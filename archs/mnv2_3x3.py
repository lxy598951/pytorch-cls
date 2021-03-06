# taken from https://github.com/tonylins/pytorch-mobilenet-v2/
# Published by Ji Lin, tonylins
# licensed under the  Apache License, Version 2.0, January 2004

import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d
# from maskrcnn_benchmark.layers import FrozenBatchNorm2d as BatchNorm2d
# from maskrcnn_benchmark.layers import Conv2d


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class PoolingBlock(nn.Module):
    def __init__(self, inp, oup):
        super(PoolingBlock, self).__init__()
        ''' Here I make modify the layers:
            Use 3 x 3 kernel, padding = 1 and stride = 2
            to replace maxpooling layer and 1 x 1 convs.
        '''
        self.block = nn.Sequential(
            # # 2 x 2 max pooling
            # nn.MaxPool2d((2, 2)),
            Conv2d(inp, oup, 3, 2, 1, bias=False),
            BatchNorm2d(oup)
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_3x3(nn.Module):
    """
    Should freeze bn
    """
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_3x3, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building pooling layers
        # 112 56 28 14 7, total five layers
        pooling_block_setting = [
            [3, 16],
            [16, 24],
            [24, 32],
            [32, 96],
            [96, 320]
        ]
        self.pooling_block = []
        for inp, oup in pooling_block_setting:
            self.pooling_block.append(PoolingBlock(inp, oup))
        self.pooling_block = nn.Sequential(*self.pooling_block)

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.return_features_indices = [1, 3, 6, 13, 17]
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, last_channel))
        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, n_class)
        )

        self._initialize_weights()

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        count = 0
        x_for_pb = x
        for i, origin_layer in enumerate(self.features):
            if i in self.return_features_indices:
                # First calculate output of pooling_block
                pb_out = self.pooling_block[count](x_for_pb)
                # Then calculate output of origin backbone.
                origin_out = origin_layer(x)
                # Make eltwise operation
                x = pb_out + origin_out
                # Updata x for next pooling_block
                x_for_pb = x
                count += 1
            else:
                x = origin_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# data = torch.randn((1, 3, 224, 224))
# net = MobileNetV2_3x3()
# print(net(data))
