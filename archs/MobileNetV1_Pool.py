# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F

class MobileNetV1_Pool(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1_Pool, self).__init__()

        def pooling_block(inp, oup):
            return nn.Sequential(
                # 2 x 2 max pooling
                nn.MaxPool2d((2, 2)),
                # 1 x 1 upsampling
                nn.Conv2d(inp, oup, 1),
                nn.BatchNorm2d(oup)
            )

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.pool1 = pooling_block(3, 32)
        self.pool2 = pooling_block(32, 128)
        self.pool3 = pooling_block(128, 256)
        self.pool4 = pooling_block(256, 512)
        self.pool5 = pooling_block(512, 1024)
        self.conv1 = conv_bn(3, 32, 2)
        self.conv2 = conv_dw(32, 64, 1)
        self.conv3 = conv_dw(64, 128, 2)
        self.conv4 = conv_dw(128, 128, 1)
        self.conv5 = conv_dw(128, 256, 2)
        self.conv6 = conv_dw(256, 256, 1)
        self.conv7 = conv_dw(256, 512, 2)
        self.conv8 = conv_dw(512, 512, 1)
        self.conv9 = conv_dw(512, 512, 1)
        self.conv10 = conv_dw(512, 512, 1)
        self.conv11 = conv_dw(512, 512, 1)
        self.conv12 = conv_dw(512, 512, 1)
        self.conv13 = conv_dw(512, 1024, 2)
        self.conv14 = conv_dw(1024, 1024, 1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        forPool = x
        # Pool1
        x = self.conv1(x)
        forPool = self.pool1(forPool)
        x = x + forPool
        # Pool1 over
        x = self.conv2(x)
        # Pool2
        x = self.conv3(x)
        forPool = self.pool2(forPool)
        x = x + forPool
        # Pool2 over
        x = self.conv4(x)
        # Pool3
        x = self.conv5(x)
        forPool = self.pool3(forPool)
        x = x + forPool
        # Pool3 over
        x = self.conv6(x)
        # Pool4
        x = self.conv7(x)
        forPool = self.pool4(forPool)
        x = x + forPool
        # Pool4 over
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        # Pool5
        x = self.conv13(x)
        forPool = self.pool5(forPool)
        x = x + forPool
        # Pool5 over
        x = self.conv14(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x