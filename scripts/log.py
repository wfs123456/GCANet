# -*- coding: utf-8 -*-
# @Time    : 2/16/20 1:48 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : log.py
# @Software: PyCharm

from config import *
import os


save_path = os.path.join("models", SAVE_PATH, "log.txt")


def my_print(str_log,save_path=save_path):
    if not isinstance(str_log,str):
        str_log = str(str_log)
    with open(save_path, "a+") as f:
        f.write(str_log + "\n")
    print(str_log)

def print_train_log(i,timer,epoch_loss, epoch_ssimloss, epoch_Mseloss, length):
    if LOSS_F == "MSE":
        loss_log = "train [%d/%d] timer %.4f, mseloss %.4f"\
                   %(i,MAX_EPOCH,timer,epoch_loss/length)
    elif LOSS_F == "L1":
        loss_log = "train [%d/%d] timer %.4f, maeloss %.4f"\
                   %(i,MAX_EPOCH,timer,epoch_loss/length)
    elif LOSS_F == "pssloss":
        loss_log = "train [%d/%d] timer %.4f, loss %.4f, pyssimloss %.4f, mseloss %.4f" \
                   %(i,MAX_EPOCH,timer,epoch_loss/length,epoch_ssimloss/length,epoch_Mseloss/length)
    elif LOSS_F == "pssloss2":
        loss_log = "train [%d/%d] timer %.4f, loss %.4f, pyssimloss %.4f, maeloss %.4f" \
                   %(i,MAX_EPOCH,timer,epoch_loss/length,epoch_ssimloss/length,epoch_Mseloss/length)
    elif LOSS_F == "cosineloss":
        loss_log = "train [%d/%d] timer %.4f, loss %.4f, cosineloss %.4f, mseloss %.4f" \
                   % (i, MAX_EPOCH, timer, epoch_loss / length, epoch_ssimloss / length, epoch_Mseloss / length)
    else:
        loss_log = "train [%d/%d] timer %.4f, mseloss %.4f" \
                   % (i, MAX_EPOCH, timer, epoch_loss / length)

    my_print(loss_log)

