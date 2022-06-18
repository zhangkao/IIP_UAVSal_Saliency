import torch
import torch.nn as nn

from torchvision.models.vgg import *
from torchvision.models.resnet import *
from torchvision.models.mobilenet import *

from torchvision.models.resnet import __all__ as resnet_name
from torchvision.models.mobilenet import __all__ as mobilenet_name
from torchvision.models.vgg import __all__ as vgg_name

__all__ = ['ReVGG', 'ReMobileNetV2', 'ReResNet']


feature_loader = {
    'vgg16': vgg16,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'mobilenet_v2': mobilenet_v2,

}

vgg_loader = {
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
}

resnet_loader = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d':resnext50_32x4d,
    'resnext101_32x8d':resnext101_32x8d,
    'wide_resnet50_2':wide_resnet50_2,
    'wide_resnet101_2':wide_resnet101_2
}

class ReMobileNetV2(nn.Module):
    def __init__(self, name='mobilenet_v2'):
        super(ReMobileNetV2, self).__init__()

        # only name = 'mobilenet_v2'
        if name not in mobilenet_name:
            raise ValueError
        if name not in feature_loader.keys():
            raise NotImplementedError

        net = feature_loader[name.lower()](pretrained=True)
        self.features = net.features

    def forward(self, x):
        x1 = self.features[0:2](x)
        x2 = self.features[2:4](x1)
        x3 = self.features[4:7](x2)
        x4 = self.features[7:14](x3)
        x5 = self.features[14:18](x4)
        # x6 = self.features[18:](x5)
        return x1, x2, x3, x4, x5


class ReResNet(nn.Module):
    def __init__(self, name='resnet50'):
        super(ReResNet, self).__init__()
        if name not in resnet_name:
            raise ValueError
        if name not in feature_loader.keys():
            raise NotImplementedError

        net = feature_loader[name.lower()](pretrained=True)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4


class ReVGG(nn.Module):
    def __init__(self, name='vgg16'):
        super(ReVGG, self).__init__()

        if name not in vgg_name:
            raise ValueError
        if name not in feature_loader.keys():
            raise NotImplementedError

        net = feature_loader[name.lower()](pretrained=True)
        self.features = net.features

    def forward(self, x):

        max_point = [i for m,i in zip(self.features.modules(),range(100)) if isinstance(m, nn.MaxPool2d)]
        mp= max_point[0:5]

        x1 = self.features[0:mp[0]](x)
        x2 = self.features[mp[0]:mp[1]](x1)
        x3 = self.features[mp[1]:mp[2]](x2)
        x4 = self.features[mp[2]:mp[3]](x3)
        x5 = self.features[mp[3]:](x4)
        return x1, x2, x3, x4, x5


if __name__ == '__main__':

    import PIL.Image as Img
    from torchsummary import summary
    from torchvision import transforms

    print("test model")

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Img.open("E:/zk/COCO_test2014_000000000063.jpg").convert('RGB')
    img = img_transform(img).unsqueeze(0)


    model = ReMobileNetV2()
    summary(model, (3, 480, 640), batch_size=8, device="cpu")

    model = resnet34(pretrained=True).cuda()
    model.eval()
    out = model(img.cuda())
    for i in range(len(out)):
        print(out[i].shape)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(out.data.cpu().numpy())
    plt.show()
