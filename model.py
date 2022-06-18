import torch
import torch.nn as nn
import torch.nn.functional as F

import os, math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model_feature import ReVGG, ReMobileNetV2, ReResNet
from model_convlstm import *


feature_loader = {
    'vgg16': ReVGG,
    'resnet18': ReResNet,
    'resnet34': ReResNet,
    'resnet50': ReResNet,
    'resnet101': ReResNet,
    'resnet152': ReResNet,
    'mobilenet_v2': ReMobileNetV2,
}


feature_inplanes = {
    'vgg16':        [128,256,512,512],
    'resnet18':     [64,128,256,512],
    'resnet34':     [64,128,256,512],
    'resnet50':     [256,512,1024,2048],
    'resnet101':    [256,512,1024,2048],
    'resnet152':    [256,512,1024,2048],
    'mobilenet_v2': [24,32,96,320],
}

init_func = {
    'uniform':nn.init.uniform_,
    'normal':nn.init.normal_,
    'constant':nn.init.constant_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'sparse': nn.init.sparse_,
    'ones':nn.init.ones_,
    'zeros':nn.init.zeros_,
}

def init_weights(model, funcname='kaiming_normal',**kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init_func[funcname](m.weight,**kwargs)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

################################################################
# Convolution functions
################################################################
class BasicConv2d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = dilation * (kernel_size - 1) // 2
        super(BasicConv2d, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class dwBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, expand_ratio=6, dilation=1, res_connect=None):
        super(dwBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup
        if res_connect is not None:
            self.use_res_connect = res_connect and self.use_res_connect

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(BasicConv2d(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            BasicConv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


################################################################
# 1 Basic Feature Extraction Sub-Models: SRF-Net
# Saliency related feature sub-network
################################################################
class uavsal_srfnet_aspp(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 planes = [64, 64, 128, 256], last_channel = 256):
        super(uavsal_srfnet_aspp, self).__init__()

        if last_channel == 128:
            planes = [32, 32, 64, 128]

        inplanes = feature_inplanes[cnn_type.lower()]

        self.conv_lv3 = BasicConv2d(inplanes[1], planes[1], 1)
        self.conv_lv4 = BasicConv2d(inplanes[2], planes[2], 1)

        aspp_rates = [6, 12, 18]
        self.lv5_aspp1 = BasicConv2d(inplanes[3], planes[3], 1)
        self.lv5_aspp2 = dwBlock(inplanes[3], planes[3], 3, dilation=aspp_rates[0])
        self.lv5_aspp3 = dwBlock(inplanes[3], planes[3], 3, dilation=aspp_rates[1])
        self.lv5_aspp4 = dwBlock(inplanes[3], planes[3], 3, dilation=aspp_rates[2])
        self.conv_lv5 = BasicConv2d(planes[3] * 4, planes[3], 1)

        inp_last = planes[1] + planes[2] + planes[3]
        self.conv_last = BasicConv2d(inp_last, last_channel, 3)

        init_weights(self,'kaiming_normal',mode='fan_out')

        self.features = feature_loader[cnn_type.lower()](name=cnn_type.lower())
        # for p in self.feature.parameters():
        # 	p.requires_grad = False

    def forward(self, x):
        _, _, c3, c4, c5 = self.features(x)

        a1 = self.lv5_aspp1(c5)
        a2 = self.lv5_aspp2(c5)
        a3 = self.lv5_aspp3(c5)
        a4 = self.lv5_aspp4(c5)
        x_c5 = torch.cat((a1, a2, a3, a4), dim=1)
        x_c5 = self.conv_lv5(x_c5)

        x_c4 = self.conv_lv4(c4)
        x_c3 = self.conv_lv3(c3)

        x_c5 = F.interpolate(x_c5, size=c3.size()[2:], mode='bilinear', align_corners=True)
        x_c4 = F.interpolate(x_c4, size=c3.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat((x_c5, x_c4, x_c3), dim=1)
        out = self.conv_last(out)

        return out

################################################################
# 2 STBlock functions
################################################################
class spConv(nn.Module):
    def __init__(self, inplanes, planes=256, kernel_size=3, stride=1, expand_ratio=6, dilation=1, res_connect=False):
        super(spConv, self).__init__()
        self.spconv = dwBlock(inplanes, planes, kernel_size, stride, expand_ratio, dilation, res_connect)
        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):
        x = self.spconv(x)
        return x

class teConv_sub(nn.Module):
    def __init__(self, inplanes, planes=256, time_dims=8, reduction=8, res_connect=False):
        super(teConv_sub, self).__init__()

        self.time_dims = time_dims
        self.res_connect = res_connect and inplanes == planes

        width = planes // reduction
        self.reduce_conv = BasicConv2d(inplanes, width, 1)
        self.sub_conv = dwBlock(2*width, width, 3, res_connect=False)

        self.last_conv = BasicConv2d(width, planes, 1)

        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        B_D, C, H, W = x.size()
        B = B_D // self.time_dims
        x1 = self.reduce_conv(x)

        sub_features = [torch.cat([x1[1:2] - x1[0:1], x1[0:1] - x1[1:2]], 1)]
        for i in range(1, B_D - 1):
            x_sub = torch.cat([x1[i:i + 1] - x1[i - 1:i], x1[i:i + 1] - x1[i + 1:i + 2]], 1)
            sub_features.append(x_sub)
        sub_features.append(torch.cat([x1[-1:] - x1[-2:-1], x1[-2:-1] - x1[-1:]], 1))

        x_sub = torch.cat(sub_features,0)
        x_sub = self.sub_conv(x_sub)

        out = self.last_conv(x_sub)

        if self.res_connect:
            out = x + out

        return out

class STBlock(nn.Module):
    def __init__(self, inplanes, planes=256, time_dims=8,
                 fu_type='sum', res_connect=True,**kwargs):
        super(STBlock, self).__init__()
        assert fu_type.lower() in ['sum','cat']

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.fu_type = fu_type.lower()
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_sp = spConv(inplanes, planes, res_connect=False)
        self.stconv_te = teConv_sub(inplanes, planes, time_dims, res_connect=False,**kwargs)

        if self.fu_type == 'sum':
            last_channel = planes
        if self.fu_type == 'cat' :
            last_channel = 2 * planes

        self.stconv_last = BasicConv2d(last_channel, planes, 1)


        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        x_sp = self.stconv_sp(x)
        x_te = self.stconv_te(x)

        if self.fu_type == 'sum':
            out = x_sp + x_te
        else:
            out = torch.cat([x_sp,x_te],dim=1)
        out = self.stconv_last(out)

        if self.res_connect:
            return x + out
        else:
            return out

################################################################
# 3 UAV saliency Models: UAVSal
################################################################
class UAVSal(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 bias_type = [1,1,1],
                 iosize=[360, 640, 45, 80],
                 planes=256,
                 pre_model_path=''):
        super(UAVSal, self).__init__()

        self.time_dims = time_dims

        # 3.1. Spatial-Temporal Feature Sub-Network
        # 1) Saliency related feature sub-network (SRF-Net) layers
        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        # 2) Spatial-Temporal Feature Block layers
        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STBlock(planes, planes, time_dims = time_dims, reduction=planes//32, res_connect = True))

        self.st_layer = nn.Sequential(*st_layers)
        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        # 3.2. Multi-Priors Sub-Network layers
        self.use_gauss_prior = bias_type[0]
        self.use_ob_prior = bias_type[1]
        self.use_context_prior = bias_type[2]
        self.num_cb = np.sum(np.array(bias_type) > 0)
        cb_inplanes = planes
        cb_ouplanes = [64,64,64]
        cb_last_channels = planes // 4

        if self.use_gauss_prior:
            nb_gaussian = 8
            cb_planes = cb_ouplanes[0]
            self.gauss_cb_layer = nn.Sequential(
                dwBlock(nb_gaussian, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.gauss_cb_layer)

        if self.use_ob_prior:
            nb_ob = 20
            cb_planes = cb_ouplanes[1]
            self.ob_cb_layer = nn.Sequential(
                dwBlock(nb_ob, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.ob_cb_layer)

        if self.use_context_prior:
            nb_channels = cb_inplanes
            cb_planes = cb_ouplanes[2]
            self.cxt_cb_prior = nn.Sequential(
                dwBlock(nb_channels, cb_planes, kernel_size=3, stride=2),
                dwBlock(cb_planes, cb_planes, kernel_size=3, stride=2),
            )
            init_weights(self.cxt_cb_prior)

        if self.num_cb:
            nb_channels = np.sum(np.array(bias_type)*np.array(cb_planes))
            self.fucb_layer = nn.Sequential(
                dwBlock(nb_channels, cb_last_channels, kernel_size=3),
            )
            self.fucbst_layer = nn.Sequential(
                dwBlock(cb_inplanes+cb_last_channels, cb_inplanes,kernel_size=3),
            )

        # 3.3. Temporal Weighted Average Sub-Network layers
        _, _, shape_r_out, shape_c_out = iosize
        self.rnn = ConvTWA((shape_r_out, shape_c_out), planes, planes, kernel_size=(3,3), num_layers=1,
                                    batch_first=True, bias=False, return_all_layers=False)

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path,map_location=device).state_dict(), strict=False)

    def forward(self, x, cb, in_state):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        if self.num_cb:
            cb_fu = []
            if self.use_gauss_prior:
                cb_gauss = self.gauss_cb_layer(cb[0])
                cb_fu.append(cb_gauss)
            if self.use_ob_prior:
                cb_ob = self.ob_cb_layer(cb[1])
                cb_fu.append(cb_ob)
            if self.use_context_prior:
                B_D, C, H, W = x.size()
                B = B_D // self.time_dims
                x_cb = x.contiguous().view(B, self.time_dims, C, H, W)
                x_cb = torch.sum(x_cb, dim=1)
                cb_cxt = self.cxt_cb_prior(x_cb)
                cb_cxt = F.interpolate(cb_cxt, size=(H, W), mode='bilinear', align_corners=True)
                cb_cxt = cb_cxt.repeat(self.time_dims, 1, 1, 1)
                cb_fu.append(cb_cxt)
            cb_fu = torch.cat(cb_fu, dim=1)
            x_cb = self.fucb_layer(cb_fu)
            x = self.fucbst_layer(torch.cat([x, x_cb], dim=1))

        B_D, C, H, W = x.size()
        x = x.contiguous().view(1, B_D, C, H, W)
        x, x_state = self.rnn(x, in_state)
        x = x.contiguous().view(B_D, x.size(2), H, W)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)
        out = F.relu(out)

        return out, x_state




################################################################
# Functions for ablation analysis
################################################################
class BasicConv3d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = dilation * (kernel_size - 1) // 2
        super(BasicConv3d, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU6(inplace=True)
        )

class STBlock_s2t(nn.Module):
    def __init__(self, inplanes, planes=256, time_dims=8,
                 res_connect=True,**kwargs):
        super(STBlock_s2t, self).__init__()

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_sp = spConv(inplanes, planes, res_connect=False)
        self.stconv_te = teConv_sub(planes, planes, time_dims, res_connect=False,**kwargs)

        self.stconv_last = BasicConv2d(planes, planes, 1)

        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        x_sp = self.stconv_sp(x)
        x_te = self.stconv_te(x_sp)

        out = self.stconv_last(x_te)

        if self.res_connect:
            return x + out
        else:
            return out

class STBlock_t2s(nn.Module):
    def __init__(self, inplanes, planes=256, time_dims=8,
                 res_connect=True,**kwargs):
        super(STBlock_t2s, self).__init__()

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_te = teConv_sub(inplanes, planes, time_dims, res_connect=False, **kwargs)
        self.stconv_sp = spConv(planes, planes, res_connect=False)

        self.stconv_last = BasicConv2d(planes, planes, 1)

        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        x_te = self.stconv_te(x)
        x_sp = self.stconv_sp(x_te)

        out = self.stconv_last(x_sp)

        if self.res_connect:
            return x + out
        else:
            return out

class STBlock_s_s2t(nn.Module):
    def __init__(self, inplanes, planes=256, time_dims=8,
                 res_connect=True,**kwargs):
        super(STBlock_s_s2t, self).__init__()

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_sp = spConv(inplanes, planes, res_connect=False)
        self.stconv_te = teConv_sub(planes, planes, time_dims, res_connect=False,**kwargs)

        self.stconv_last = BasicConv2d(planes, planes, 1)

        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        x_sp = self.stconv_sp(x)
        x_te = self.stconv_te(x_sp)

        x_te = x_sp + x_te
        out = self.stconv_last(x_te)

        if self.res_connect:
            return x + out
        else:
            return out

class STC3D(nn.Module):
    def __init__(self, inplanes=256, planes=256, time_dims=5,
                 kernel_size=3, stride=1, dilation=1, res_connect=True):
        super(STC3D, self).__init__()

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_te = BasicConv3d(inplanes, planes, kernel_size, stride, dilation)

        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        B_D, C, H, W = x.size()
        B = B_D // self.time_dims
        x_te = x.contiguous().view(B, self.time_dims, C, H, W)
        x_te = x_te.permute(0, 2, 1, 3, 4)
        x_te = self.stconv_te(x_te)
        x_te = x_te.permute(0, 2, 1, 3, 4)
        out = x_te.contiguous().view(B_D, x_te.size(2), H, W)

        if self.res_connect:
            return x + out
        else:
            return out

class STC2_3D(nn.Module):
    def __init__(self, inplanes=256, planes=256, time_dims=5, kernel_size=3,
                 fu_type='sum', res_connect=True,**kwargs):
        super(STC2_3D, self).__init__()
        assert fu_type.lower() in ['sum','cat']

        self.res_connect = res_connect and inplanes == planes
        self.time_dims = time_dims
        self.fu_type = fu_type.lower()
        self.inplanes = inplanes
        self.planes = planes

        self.stconv_sp = BasicConv2d(inplanes, planes, kernel_size)
        self.stconv_te = BasicConv3d(inplanes, planes, kernel_size)

        if self.fu_type == 'sum':
            last_channel = planes
        if self.fu_type == 'cat' :
            last_channel = 2 * planes

        self.stconv_last = BasicConv2d(last_channel, planes, 1)


        init_weights(self, 'kaiming_normal', mode='fan_out')

    def forward(self, x):

        x_sp = self.stconv_sp(x)

        B_D, C, H, W = x.size()
        B = B_D // self.time_dims
        x_te = x.contiguous().view(B, self.time_dims, C, H, W)
        x_te = x_te.permute(0, 2, 1, 3, 4)
        x_te = self.stconv_te(x_te)
        x_te = x_te.permute(0, 2, 1, 3, 4)
        x_te = x_te.contiguous().view(B_D, x_te.size(2), H, W)


        if self.fu_type == 'sum':
            out = x_sp + x_te
        else:
            out = torch.cat([x_sp,x_te],dim=1)
        out = self.stconv_last(out)

        if self.res_connect:
            return x + out
        else:
            return out

################################################################
# Ablation Study Models
# For Table I:
# 1 Sp-Net: UAVSAL_SpCOnv --> out
# 2 Te-Net: UAVSAL_teConv --> out
# 3 ST-Net: UAVSAL_STBlocks --> out (num_stblock=2)
# 4 MP-Net: UAVSAL_MP --> out (bias_type = [1,1,1])
# 5 Final (ST+MP+TWA): UAVSal --> out
#
# For Table II:
# UAVSAL_STBlocks -->  num_stblock in [1,2,3,4]
#
# For Table III:
# 1 UAVSAL_STC3D
# 2 UAVSAL_STBlocks_type --> st_type in ['st', 's2t', 't2s']
#
# For Table IV:
# UAVSAL_MP -->  change bias_type, e.g. [1,1,1],[1,1,0],[1,0,0]...
#
# For Table V:
# UAVSALt_LSTM -->  out
################################################################

# Table1: Sp-Net and Te-Net
class UAVSAL_SpCOnv(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 num_stblock=2,
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_SpCOnv, self).__init__()

        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(dwBlock(planes, planes, 3, res_connect=True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path, map_location=device).sfnet.state_dict(), strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

class UAVSAL_teConv(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_teConv, self).__init__()

        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []

        for i in range(self.num_stblock):
            st_layers.append(teConv_sub(planes, planes, time_dims, reduction=planes//32, res_connect=True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path,map_location=device).sfnet.state_dict(),strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

### Table1: ST-Net: num_stblock=2
### Table2: num_stblock in [1,2,3,4]
class UAVSAL_STBlocks(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_STBlocks, self).__init__()

        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STBlock(planes, planes, time_dims = time_dims, reduction=planes//32, res_connect = True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path,map_location=device).sfnet.state_dict(),strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out, x

### Table3: st_type in ['st', 's2t', 't2s']
class UAVSAL_STBlocks_type(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 planes=256,
                 st_type='st',
                 pre_model_path=''):
        super(UAVSAL_STBlocks_type, self).__init__()

        assert st_type.lower() in ['st', 's2t', 't2s', 's_s2t']
        if st_type.lower() in ['s2t']:
            STBlock_type = STBlock_s2t
        elif st_type.lower() in ['t2s']:
            STBlock_type = STBlock_t2s
        elif st_type.lower() in ['s_s2t']:
            STBlock_type = STBlock_s_s2t
        else:
            STBlock_type = STBlock

        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STBlock_type(planes, planes, time_dims = time_dims, reduction=planes//32, res_connect = True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path,map_location=device).sfnet.state_dict(),strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

class UAVSAL_STC3D(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_STC3D, self).__init__()

        self.time_dims = time_dims
        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STC3D(planes, planes, kernel_size=3, time_dims= time_dims, res_connect=True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path,map_location=device).sfnet.state_dict(),strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

class UAVSAL_STC2_3D(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_STC2_3D, self).__init__()

        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STC2_3D(planes, planes, time_dims = time_dims, res_connect = True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.sfnet.load_state_dict(torch.load(pre_model_path,map_location=device).sfnet.state_dict(),strict=False)

    def forward(self, x):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

### Table1: MP-Net: bias_type = [1,1,1]
### Table4: change bias_type
class UAVSAL_MP(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 bias_type = [1,1,1],
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_MP, self).__init__()

        self.time_dims = time_dims
        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STBlock(planes, planes, time_dims=time_dims, reduction=planes // 32, res_connect=True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        ## To Do MP-Net layers
        # three kinds of bias prior maps
        # 1. gaussion maps, e.g. 8 gaussion maps with different variance
        # 2. observed maps, e.g 20 maps from mean value of each GT videos
        # 3. learned from the sences, e.g. one time_dim frames generate the bias feature maps

        self.use_gauss_prior = bias_type[0]
        self.use_ob_prior = bias_type[1]
        self.use_context_prior = bias_type[2]
        self.num_cb = np.sum(np.array(bias_type) > 0)
        cb_inplanes = planes
        cb_ouplanes = [64,64,64]
        cb_last_channels = planes // 4 #64 or 32

        if self.use_gauss_prior:
            nb_gaussian = 8
            cb_planes = cb_ouplanes[0]
            self.gauss_cb_layer = nn.Sequential(
                dwBlock(nb_gaussian, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.gauss_cb_layer)

        if self.use_ob_prior:
            nb_ob = 20
            cb_planes = cb_ouplanes[1]
            self.ob_cb_layer = nn.Sequential(
                dwBlock(nb_ob, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.ob_cb_layer)

        if self.use_context_prior:
            nb_channels = cb_inplanes
            cb_planes = cb_ouplanes[2]
            self.cxt_cb_prior = nn.Sequential(
                dwBlock(nb_channels, cb_planes, kernel_size=3, stride=2),
                dwBlock(cb_planes, cb_planes, kernel_size=3, stride=2),
            )
            init_weights(self.cxt_cb_prior)

        if self.num_cb:
            nb_channels = np.sum(np.array(bias_type)*np.array(cb_planes))
            self.fucb_layer = nn.Sequential(
                dwBlock(nb_channels, cb_last_channels, kernel_size=3),
            )
            self.fucbst_layer = nn.Sequential(
                dwBlock(cb_inplanes+cb_last_channels, cb_inplanes,kernel_size=3),
            )

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained  weights")
            self.load_state_dict(torch.load(pre_model_path,map_location=device).state_dict(), strict=False)

    def forward(self, x, cb):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        if self.num_cb:
            cb_fu = []
            if self.use_gauss_prior:
                cb_gauss = self.gauss_cb_layer(cb[0])
                cb_fu.append(cb_gauss)
            if self.use_ob_prior:
                cb_ob = self.ob_cb_layer(cb[1])
                cb_fu.append(cb_ob)
            if self.use_context_prior:
                B_D, C, H, W = x.size()
                B = B_D // self.time_dims
                x_cb = x.contiguous().view(B, self.time_dims, C, H, W)
                x_cb = torch.sum(x_cb, dim=1)
                cb_cxt = self.cxt_cb_prior(x_cb)
                cb_cxt = F.interpolate(cb_cxt, size=(H, W), mode='bilinear', align_corners=True)
                cb_cxt = cb_cxt.repeat(self.time_dims, 1, 1, 1)
                cb_fu.append(cb_cxt)
            cb_fu = torch.cat(cb_fu, dim=1)
            x_cb = self.fucb_layer(cb_fu)
            x = self.fucbst_layer(torch.cat([x, x_cb], dim=1))

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out

### Table5: LSTM
class UAVSAL_LSTM(nn.Module):
    def __init__(self, cnn_type='mobilenet_v2',
                 time_dims=5,
                 num_stblock=2,
                 bias_type = [1,1,1],
                 iosize=[360, 640, 45, 80],
                 planes=256,
                 pre_model_path=''):
        super(UAVSAL_LSTM, self).__init__()

        self.time_dims = time_dims
        self.sfnet = uavsal_srfnet_aspp(cnn_type,last_channel=planes)

        self.num_stblock = num_stblock
        st_layers = []
        for i in range(self.num_stblock):
            st_layers.append(STBlock(planes, planes, time_dims = time_dims, reduction=planes//32, res_connect = True))

        self.st_layer = nn.Sequential(*st_layers)

        self.fust_layer = nn.Sequential(
            dwBlock(planes, planes, kernel_size=3),
        )

        self.use_gauss_prior = bias_type[0]
        self.use_ob_prior = bias_type[1]
        self.use_context_prior = bias_type[2]
        self.num_cb = np.sum(np.array(bias_type) > 0)
        cb_inplanes = planes
        cb_ouplanes = [64,64,64]
        cb_last_channels = planes // 4

        if self.use_gauss_prior:
            nb_gaussian = 8
            cb_planes = cb_ouplanes[0]
            self.gauss_cb_layer = nn.Sequential(
                dwBlock(nb_gaussian, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.gauss_cb_layer)

        if self.use_ob_prior:
            nb_ob = 20
            cb_planes = cb_ouplanes[1]
            self.ob_cb_layer = nn.Sequential(
                dwBlock(nb_ob, cb_planes, kernel_size=3),
                dwBlock(cb_planes, cb_planes, kernel_size=3),
            )
            init_weights(self.ob_cb_layer)

        if self.use_context_prior:
            nb_channels = cb_inplanes
            cb_planes = cb_ouplanes[2]
            self.cxt_cb_prior = nn.Sequential(
                dwBlock(nb_channels, cb_planes, kernel_size=3, stride=2),
                dwBlock(cb_planes, cb_planes, kernel_size=3, stride=2),
            )
            init_weights(self.cxt_cb_prior)

        if self.num_cb:
            nb_channels = np.sum(np.array(bias_type)*np.array(cb_planes))
            self.fucb_layer = nn.Sequential(
                dwBlock(nb_channels, cb_last_channels, kernel_size=3),
            )
            self.fucbst_layer = nn.Sequential(
                dwBlock(cb_inplanes+cb_last_channels, cb_inplanes,kernel_size=3),
            )

        _, _, shape_r_out, shape_c_out = iosize
        self.rnn = ConvLSTM((shape_r_out, shape_c_out), planes, planes, kernel_size=(3,3), num_layers=1,
                                    batch_first=True, bias=False, return_all_layers=False)

        self.conv_out_st = dwBlock(planes, 1, kernel_size=3)

        init_weights(self.st_layer,   'kaiming_normal', mode='fan_out')
        init_weights(self.fust_layer, 'kaiming_normal', mode='fan_out')
        init_weights(self.conv_out_st,'kaiming_normal', mode='fan_out')

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path,map_location=device).state_dict(), strict=False)

    def forward(self, x, cb, in_state):
        x = self.sfnet(x)
        x = self.st_layer(x)
        x = self.fust_layer(x)

        if self.num_cb:
            cb_fu = []
            if self.use_gauss_prior:
                cb_gauss = self.gauss_cb_layer(cb[0])
                cb_fu.append(cb_gauss)
            if self.use_ob_prior:
                cb_ob = self.ob_cb_layer(cb[1])
                cb_fu.append(cb_ob)
            if self.use_context_prior:
                B_D, C, H, W = x.size()
                B = B_D // self.time_dims
                x_cb = x.contiguous().view(B, self.time_dims, C, H, W)
                x_cb = torch.sum(x_cb, dim=1)
                cb_cxt = self.cxt_cb_prior(x_cb)
                cb_cxt = F.interpolate(cb_cxt, size=(H, W), mode='bilinear', align_corners=True)
                cb_cxt = cb_cxt.repeat(self.time_dims, 1, 1, 1)
                cb_fu.append(cb_cxt)
            cb_fu = torch.cat(cb_fu, dim=1)
            x_cb = self.fucb_layer(cb_fu)
            x = self.fucbst_layer(torch.cat([x, x_cb], dim=1))

        B_D, C, H, W = x.size()
        x = x.contiguous().view(1, B_D, C, H, W)
        x, x_state = self.rnn(x, in_state)
        x = x.contiguous().view(B_D, x.size(2), H, W)

        out = self.conv_out_st(x)
        out = torch.sigmoid(out)

        return out, x_state

