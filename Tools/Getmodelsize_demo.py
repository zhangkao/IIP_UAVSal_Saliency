import torch, os, cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math, shutil, copy


from model import UAVSal


def getModelSize(model):
    param_size = 0
    param_sum = 0
    param_trainable = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

        if param.requires_grad:
            param_trainable += param.nelement() * param.element_size()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    param_mb = param_size / 1024 / 1024
    buffer_mb = buffer_size / 1024 / 1024
    all_mb = (param_size + buffer_size) / 1024 / 1024

    # train_p_mb = param_trainable / 1024 / 1024

    print('param  size：{:.2f} MB'.format(param_mb))
    print('buffer size：{:.2f} MB'.format(buffer_mb))
    print('total  size：{:.2f} MB'.format(all_mb))

    # print('trainable params size：{:.2f} MB'.format(train_p_mb))

    return param_mb#,buffer_mb,all_mb


if __name__ == '__main__':

    other_size = 0

    model = UAVSal()
    print('\nmodel')
    model_size = getModelSize(model)

    print('\nSRF-Net')
    srf_size = getModelSize(model.sfnet)

    print('\nST-Blocks')
    stblocks_size = getModelSize(model.st_layer)

    print('\nfust_layer')
    fust_size = getModelSize(model.fust_layer)
    other_size += fust_size

    mp_size = 0
    print('\ngauss_cb_layer')
    mp_size += getModelSize(model.gauss_cb_layer)

    print('\nob_cb_layer')
    mp_size += getModelSize(model.ob_cb_layer)

    print('\ncxt_cb_prior')
    mp_size += getModelSize(model.cxt_cb_prior)

    print('\nfucb_layer')
    mp_size += getModelSize(model.fucb_layer)

    print('\nfucbst_layer')
    mp_size += getModelSize(model.fucbst_layer)

    print('\nrnn')
    twa_size = getModelSize(model.rnn)

    print('\nconv_out_st')
    other_size += getModelSize(model.conv_out_st)

    print('\n\n-----Params Size-----')
    print('Total UAVSal: %.2f MB' % model_size)
    print('SRF-Net: %.2f MB' % srf_size)
    print('ST-Blocks: %.2f MB' % stblocks_size)
    print('MP-Net: %.2f MB' % mp_size)
    print('TWA-Net: %.2f MB' % twa_size)
    print('Other: %.2f MB' % other_size)
    print('diff: %.2f MB' % (model_size-srf_size-stblocks_size-mp_size-twa_size-other_size))

    print('\nTotal UAVSal (param+buffer)): 51.59 MB')
    print('Saved UAVSal model file size: 51.87 MB')

    print('done')
