# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : model.py
# @Software : PyCharm

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import os

def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    # in_channels = 3
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

def conv(in_channel, out_channel, kernel_size, dilation=1, bn=True):
    #padding = 0
    # if kernel_size % 2 == 1:
    #     padding = int((kernel_size - 1) / 2)
    padding = dilation # maintain the previous size
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            # nn.BatchNorm2d(out_channel, momentum=0.005),
            nn.ReLU(inplace=True)
        )

# class Inception(nn.Module):
#     def __init__(self,in_channel):
#         super(Inception, self).__init__()
#         self.conv1x1 = nn.Sequential(nn.Conv2d(in_channel,64,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.conv2x1 = nn.Sequential(nn.Conv2d(in_channel,64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.conv3x1 = nn.Sequential(nn.Conv2d(in_channel, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
#         self.conv4x1 = nn.Sequential(nn.Conv2d(in_channel, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
#
#         self.conv2x2 = nn.Sequential(nn.Conv2d(64,64,3,padding=2,dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.conv3x2 = nn.Sequential(nn.Conv2d(128,128,5,padding=4,dilation=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
#         self.conv4x2 = nn.Sequential(nn.Conv2d(256,256,7,padding=6,dilation=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
#         self.init_param()
#     def forward(self, x):
#         x1 = self.conv1x1(x)
#         x2 = self.conv2x2(self.conv2x1(x))
#         x3 = self.conv3x2(self.conv3x1(x))
#         x4 = self.conv4x2(self.conv4x1(x))
#         x = torch.cat((x1,x2,x3,x4),1)
#         return x
#
#     def init_param(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

class AttenModule(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AttenModule, self).__init__()

        self.attention = nn.Sequential(nn.Conv2d(in_channel, 1, 3,padding=1, bias=True),
                                       nn.Sigmoid()
                                       )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU(inplace=True),
                                       )

        self.init_param()
    def forward(self, x):
        atten = self.attention(x)
        features = self.conv1(x)
        x = features * atten
        return x, atten

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SPPSELayer(nn.Module):
    def __init__(self,in_cahnnel, channel, reduction=16):
        super(SPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(in_cahnnel*21, in_cahnnel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_cahnnel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(in_cahnnel, channel, 1),
                                   nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size() # b: number; c: channel;
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b,out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        x = self.conv2(x)
        y = y * x
        return y

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # cfg1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        cfg2 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.front_end1 = make_layers(cfg1, 3, batch_norm=True)
        self.front_end2 = make_layers(cfg2, 3, batch_norm=True)

        # self.Inception1 = Inception(512)
        self.attenModule1 = AttenModule(512, 256)
        self.attenModule2 = AttenModule(256, 128)
        self.attenModule3 = AttenModule(128, 64)

        self.SPPSEMoudule1 = SPPSELayer(512,256)
        self.SPPSEMoudule2 = SPPSELayer(256, 128)
        self.SPPSEMoudule3 = SPPSELayer(128, 64)

        self.ReduConv1 = conv(512, 256, 3, dilation=1)
        self.ReduConv2 = conv(256, 128, 3, dilation=2)
        self.ReduConv3 = conv(128, 64, 3, dilation=3)
        # self.A_conv =  nn.Sequential(conv(512, 128, 3), conv(128, 64, 3))
        # self.final = nn.Sequential(nn.Conv2d(64, 1, 1), nn.ReLU())
        self.final = nn.Conv2d(64,1,1)

        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("loading pretrained vgg16_bn!")
        if os.path.exists("/home/liuqi/PycharmProjects/SPPSENet/weights/vgg16_bn.pth"):
            print("find pretrained weights!")
            vgg16_bn = models.vgg16_bn(pretrained=False)
            vgg16_weights = torch.load("/home/liuqi/PycharmProjects/SPPSENet/weights/vgg16_bn.pth")
            vgg16_bn.load_state_dict(vgg16_weights)
        else:
            vgg16_bn = models.vgg16_bn(pretrained=True)
        # the front conv block's parameter no training
        # for p in self.front_end1.parameters():
        #     p.requires_grad = False

        # self.front_end1.load_state_dict(vgg16_bn.features[:23].state_dict())
        self.front_end2.load_state_dict(vgg16_bn.features[:33].state_dict())

    def forward(self, x, vis=False):
        # y = self.front_end1(x)

        #dense block
        x = self.front_end2(x)

        x1,atten1 = self.attenModule1(x)
        y1 = self.SPPSEMoudule1(x)
        x = torch.cat((x1,y1), 1)
        x = self.ReduConv1(x)


        x2,atten2 = self.attenModule2(x)
        y2 = self.SPPSEMoudule2(x)
        x = torch.cat((x2, y2), 1)
        x = self.ReduConv2(x)
        
        x3, atten3 = self.attenModule3(x)
        y3 = self.SPPSEMoudule3(x)
        x = torch.cat((x3, y3), 1)
        x = self.ReduConv3(x)

        x = self.final(x)
        # att = F.interpolate(att, scale_factor=8, mode="nearest", align_corners=None)
        # x = F.interpolate(x, scale_factor=8, mode="nearest", align_corners=None)
        if vis:
            return x, atten1, atten2, atten3
        return x


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net = Net()
    # print(net.front_end.state_dict())
    x = torch.ones((16, 3, 128, 128))
    print(x.size())
    y= net(x)
    print(y.size())