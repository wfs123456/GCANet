# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/19 18:23
# @File     : PSNR.py
# @Software : PyCharm

import numpy
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))