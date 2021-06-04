# -*- coding: utf-8 -*-
# @Time    : 2/14/20 10:11 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : loss.py
# @Software: PyCharm

import torch.nn as nn
from torch.nn import functional as F
from scripts.ssim_loss import SSIM
from config import *

def mseloss(est_dmp, target_dmp):
    loss = nn.MSELoss(reduction='mean').cuda()(est_dmp,target_dmp)
    loss = loss / (BATCH_SIZE * CROP_NUM)
    return loss

def maeloss(est_dmp, target_dmp):
    loss = nn.L1Loss(reduction='sum').cuda()(est_dmp,target_dmp)
    return loss, False, False

def pssloss(est_dmp, target_dmp):
    pyssim_loss = 1.0 - SSIM(size_average=False).cuda()(est_dmp,target_dmp) + \
                  (1.0 - SSIM(size_average=False).cuda()(F.avg_pool2d(est_dmp, kernel_size=(2, 2)),
                                                         F.avg_pool2d(target_dmp, kernel_size=(2, 2)))) + \
                  (1.0 - SSIM(size_average=False).cuda()(F.avg_pool2d(est_dmp, kernel_size=(4, 4)),
                                                         F.avg_pool2d(target_dmp, kernel_size=(4, 4))))
    
    pyssim_loss = SSIM_WEIGHT * pyssim_loss.sum()
    mse_loss = nn.MSELoss(reduction='sum').cuda()(est_dmp,target_dmp)

    loss = pyssim_loss +mse_loss
    return loss,pyssim_loss,mse_loss


def pssloss2(est_dmp, target_dmp):
    pyssim_loss = 1.0 - SSIM(size_average=False).cuda()(est_dmp,target_dmp) + \
                  (1.0 - SSIM(size_average=False).cuda()(F.avg_pool2d(est_dmp, kernel_size=(2, 2)),
                                                         F.avg_pool2d(target_dmp, kernel_size=(2, 2)))) + \
                  (1.0 - SSIM(size_average=False).cuda()(F.avg_pool2d(est_dmp, kernel_size=(4, 4)),
                                                         F.avg_pool2d(target_dmp, kernel_size=(4, 4))))
    pyssim_loss = SSIM_WEIGHT * pyssim_loss.sum()
    mse_loss = nn.L1Loss(reduction='sum').cuda()(est_dmp,target_dmp)

    loss = pyssim_loss + mse_loss
    return loss,pyssim_loss,mse_loss

def cosineLoss(est_dmp, target_dmp):
    est_dmp_ = est_dmp.view(est_dmp.size()[0],-1)
    target_dmp_ = target_dmp.view(est_dmp.size()[0], -1)
    cosine_loss = sum(1.0-F.cosine_similarity(est_dmp_,target_dmp_)) * 5.0

    mse_loss = nn.MSELoss(reduction='sum').cuda()(est_dmp,target_dmp)

    loss = cosine_loss + mse_loss
    return loss, cosine_loss, mse_loss



def get_loss():
    if LOSS_F == "MSE":
        criterion = mseloss
    elif LOSS_F == "L1":
        criterion = maeloss
    elif LOSS_F == "pssloss":
        criterion = pssloss
    elif LOSS_F == "pssloss2":
        criterion = pssloss2
    elif LOSS_F == "cosineloss":
        criterion = cosineLoss
    else:
        print("not find %s!"%LOSS_F)
        exit(-1)
    return criterion

if __name__ == "__main__":
    import torch
    import numpy as np

    a = np.random.uniform(size=(2,1,16,16))
    b = np.random.uniform(size=(2,1,16,16))
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    print(cosineLoss(a, b))
