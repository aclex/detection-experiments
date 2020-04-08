import backbone.outlet.mobilenetv3.mobilenetv3 as outlet

from backbone.outlet.mobilenetv3.mobilenetv3 import Block, hswish

from backbone.feature_hook import FeatureHook


class MobileNetV3_Large(outlet.MobileNetV3_Large):
    def __init__(self, **kwargs):
        super(MobileNetV3_Small, self).__init__(**kwargs)

        self.c4 = FeatureHook()

        self.bneck[11].register_forward_hook(self.c4)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))

        return self.c4.output, out # C4 and C5


class MobileNetV3_Small(outlet.MobileNetV3_Small):
    def __init__(self, **kwargs):
        super(MobileNetV3_Small, self).__init__(**kwargs)

        self.c4 = FeatureHook()

        self.bneck[7].register_forward_hook(self.c4)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))

        return self.c4.output, out # C4 and C5
