from __future__ import print_function

import torch
import torch.nn as nn
import math
EPS = 2.2204e-16

def get_sum(input):
    size_h, size_w = input.shape[2:]
    v_sum = torch.sum(input, (2,3), keepdim=True)
    return v_sum.repeat(1, 1, size_h, size_w)

def get_max(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.max(torch.max(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)

def get_min(input):
    size_h, size_w = input.shape[2:]
    v_max = torch.min(torch.min(input, 2, keepdim=True)[0], 3, keepdim=True)[0]
    return v_max.repeat(1, 1, size_h, size_w)

def get_mean(input):
    size_h, size_w = input.shape[2:]
    v_mean = torch.mean(input, (2,3), keepdim=True)
    return v_mean.repeat(1, 1, size_h, size_w)

def get_std(input):
    size_h, size_w = input.shape[2:]
    # v_mean = torch.mean(input, (2,3), keepdim=True)
    # tmp = torch.sum((input-v_mean)**2,(2,3),keepdim=True) / (size_h*size_w-1)
    # return torch.sqrt(tmp).repeat(1,1, size_h, size_w)
    v_std = torch.std(input, (2, 3), keepdim=True)
    return v_std.repeat(1, 1, size_h, size_w)


def loss_kl(y_pred, y_true):
    kl_value = metric_kl(y_pred, y_true)
    loss_value = 10 * kl_value

    return torch.mean(loss_value, 0)

def loss_fu(y_pred,y_true):

    kl_value  = metric_kl(y_pred,y_true)
    cc_value  = metric_cc(y_pred, y_true)
    nss_value = metric_nss(y_pred, y_true)
    loss_value = 10 * kl_value - 2 * cc_value - nss_value

    return torch.mean(loss_value,0)

def loss_fu_dy(y_pred,y_true):
    B, D, C, H, W = y_pred.size()
    y_pred = torch.reshape(y_pred, (B * D, C, H, W))
    y_true = torch.reshape(y_true, (B * D, 2, H, W))

    kl_value  = metric_kl(y_pred,y_true)
    cc_value  = metric_cc(y_pred, y_true)
    nss_value = metric_nss(y_pred, y_true)
    loss_value = 10 * kl_value - 2 * cc_value - nss_value

    return torch.mean(loss_value,0)

def metric_kl(y_pred,y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = y_true / (get_sum(y_true) + EPS)
    y_pred = y_pred / (get_sum(y_pred) + EPS)

    return torch.mean(torch.sum(y_true * torch.log((y_true / (y_pred + EPS)) + EPS), (2,3)),0)

def metric_cc(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_mean(y_true)) / (get_std(y_true) + EPS)
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred) + EPS)

    y_true = y_true - get_mean(y_true)
    y_pred = y_pred - get_mean(y_pred)
    r1 = torch.sum(y_true * y_pred,(2,3))
    r2 = torch.sqrt(torch.sum(y_pred*y_pred,(2,3))*torch.sum(y_true*y_true,(2,3)))
    return torch.mean(r1 / (r2 +EPS) ,0)

def metric_nss(y_pred, y_true):
    y_true = y_true[:, 1:2, :, :]
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred)+ EPS)

    return torch.mean(torch.sum(y_true * y_pred, dim=(2,3)) / (torch.sum(y_true, dim=(2,3))+EPS),0)

def metric_sim(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]
    y_true = (y_true - get_min(y_true)) / (get_max(y_true) - get_min(y_true) + EPS)
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    y_true = y_true / (get_sum(y_true) + EPS)
    y_pred = y_pred / (get_sum(y_pred) + EPS)

    diff = torch.min(y_true,y_pred)
    score = torch.sum(diff,dim=(2,3))

    return torch.mean(score,0)

def loss_ml(y_pred, y_true):
    y_true = y_true[:, 0:1, :, :]

    y_pred = y_pred / (get_max(y_pred) + EPS)
    return torch.mean((y_pred - y_true)*(y_pred - y_true) / (1 - y_true + 0.1))

