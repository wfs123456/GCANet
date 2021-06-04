# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : config.py
# @Software : PyCharm

CUDA = '0'
#device_ids = [0,1,2,3,4,5,6,7]
# HOME = "D:\Project_crowd counting\dataset\ShanghaiTech"
# DATASET = "part_A_final" # ShanghaiTech/part_A_final, UCF-CC-50/folder1, UCF-QNRF-Nor
ROOT = ".\\Derain_datasets"

RESUME = False       #False
USE_VISDOM = False

CROP_NUM = 1       # 4
CROP_SIZE = 128    # 128

BATCH_SIZE = 1    # 32,32*len(device_ids)
MOMENTUM = 0.95
WEIGHT_DECAY = 5*1e-4
LEARNING_RATE = 1e-4  # 1e-6
MAX_EPOCH = 500
STEPS = [50,500]
LR_DECAY = 0.95       # 0.5, 0.794328
OPTIMIZER = "Adam"   # SGD, Adam
LOSS_F = "MSE"      # MSE, L1, pssloss, pssloss2
SSIM_WEIGHT = 1       #0.5,0.1,1

MODEL = "GCANet"

SAVE_PATH = "%s/weight/%s%sDerain"%("Rain1400",LOSS_F,str(LEARNING_RATE))

RANDOM_CROP = False       # False
RANDOM_FLIP = 0.5        # 0.5
RANDOM_2GRAY = 0.2       # 0.2
RANDOM_HUE = 0.0           # 0.2
RANDOM_SATURATION = 0.0    # 0.2
RANDOM_BRIGHTNESS = 0.0    # 0.2
RANDOM_CHANNEL = 0.0       # 0.2
RANDOM_NOISE = 0.0       # 0.2
DIVIDE = 16            # 16
PADDING = None             # 32
SEED = 19961112


VAR_LIST = ["BATCH: %d"%BATCH_SIZE, "OPTIM: %s"%OPTIMIZER, "LR: %s"%str(LEARNING_RATE),"LOSS_F: %s"%LOSS_F,
            "CUDA: %s"%CUDA,"LR_DECAY: %s"%LR_DECAY,"STEPS: %s"%STEPS,"MODEL: %s"%MODEL, "RANDOM_CROP: %s"%str(RANDOM_CROP),
            "RANDOM_FLIP: %s"%str(RANDOM_FLIP),"RANDOM_2GRAY: %s"%str(RANDOM_2GRAY),"RANDOM_HUE: %s"%str(RANDOM_HUE),
            "RANDOM_SATURATION: %s"%str(RANDOM_SATURATION),"RANDOM_BRIGHTNESS: %s"%str(RANDOM_BRIGHTNESS),
            "RANDOM_CHANNEL: %s"%str(RANDOM_CHANNEL),"RANDOM_NOISE: %s"%str(RANDOM_NOISE),"DIVIDE: %s"%DIVIDE,
            "PADDING: %s"%PADDING,"CROP_NUM: %s"%CROP_NUM, "CROP_SIZE: %s"%CROP_SIZE,"SAVE_PATH: %s"%SAVE_PATH,"SEED: %s"%str(SEED),
            "SSIM_WEIGHT: %s"%str(SSIM_WEIGHT),"root: %s"%ROOT]
