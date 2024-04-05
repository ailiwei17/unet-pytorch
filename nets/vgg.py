import torch.nn as nn
from torch.hub import load_state_dict_from_url
from nets.module import SpatialGroupEnhance


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, update=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
        self.update = update
        if update:
            self.sge_feat1 = SpatialGroupEnhance(64)
            self.sge_feat2 = SpatialGroupEnhance(128)
            self.sge_feat3 = SpatialGroupEnhance(256)
            self.sge_feat4 = SpatialGroupEnhance(512)
            self.sge_feat5 = SpatialGroupEnhance(512)


    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[:4](x)
        if self.update:
            feat1 = self.sge_feat1(feat1)
        feat2 = self.features[4:9](feat1)
        if self.update:
            feat2 = self.sge_feat2(feat2)
        feat3 = self.features[9:16](feat2)
        if self.update:
            feat3 = self.sge_feat3(feat3)
        if not self.update:
            feat4 = self.features[16:23](feat3)
            feat5 = self.features[23:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]
        else:
            return [feat1, feat2, feat3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict,strict=False)

    del model.avgpool
    del model.classifier
    return model
