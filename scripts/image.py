# -*- coding: utf-8 -*-
# @Time    : 2/12/20 11:34 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : image.py
# @Software: PyCharm


import numpy as np
import cv2
import random
import skimage.util.noise as noise


def divideByfactor(img,gt_dmap, factor=32):
    shape1,shape2 = gt_dmap.shape[0], gt_dmap.shape[1]
    crop1, crop2 = 0, 0
    if (shape1 % factor != 0):
        shape1 = int(shape1 // factor * factor)
        crop1 = random.randint(0, shape1 % factor)
    if (shape2 % factor != 0):
        shape2 = int(shape2 // factor * factor)
        crop2 = random.randint(0, shape2 % factor)
    img_ = img[crop1:shape1 + crop1, crop2:shape2 + crop2,:].copy()
    gt_dmap_ = gt_dmap[crop1:shape1 + crop1, crop2:shape2 + crop2].copy()
    return img_, gt_dmap_

def paddingByfactor(img,gt_dmap, factor=32):
    shape1,shape2 = gt_dmap.shape[0], gt_dmap.shape[1]
    w,h = shape1%factor,shape2%factor
    if (w!=0) or (h!=0):
        top,bottom,left,right = 0,0,0,0
        if w!=0:
            top = (factor-w)//2
            bottom = factor-w-top
        if h!=0:
            left = (factor-h)//2
            right = factor-h-left
        img = cv2.copyMakeBorder(img.copy(),top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
        gt_dmap = cv2.copyMakeBorder(gt_dmap.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, gt_dmap

def random_crop(img,gt_dmap, factor):
    h, w = gt_dmap.shape[0], gt_dmap.shape[1]

    if factor == 0:
        x0,x1 = 0, w//2
        y0,y1 = 0, h//2
    elif factor == 1:
        x0,x1 = w//2, w
        y0,y1 = 0, h//2
    elif factor == 2:
        x0,x1 = 0, w//2
        y0,y1 = h//2, h
    elif factor == 3:
        x0,x1 = w//2, w
        y0,y1 = h//2, h
    elif factor == 4:
        w_ = random.randint(128, w)
        x0 = random.randint(0, w-w_)
        x1 = x0+w_
        h_ = random.randint(128, h)
        y0 = random.randint(0, h-h_)
        y1 = y0+h_
    else:
        x0, x1 = 0, w
        y0, y1 = 0, h

    img_ = img[y0:y1, x0:x1, :].copy()
    gt_dmap_ = gt_dmap[y0:y1, x0:x1].copy()
    return img_, gt_dmap_

def random_flip(img, gt_dmap,probability=0.5):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        img_ = cv2.flip(img_,1)
        gt_dmap_ = cv2.flip(gt_dmap_,1)
    return img_,gt_dmap_

def random_2gray(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
        img_ = np.zeros_like(img)
        img_[:, :, 0], img_[:, :, 1], img_[:, :, 2] = gray, gray, gray
        # gray = (img_[:,:,0]).copy()*0.114+(img_[:,:,1]).copy()*0.587+(img_[:,:,2]).copy()*0.299
        # img_[:,:,0],img_[:,:,1],img_[:,:,2] = gray,gray,gray
    return img_, gt_dmap_

def random_hue(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        times = random.uniform(0.8, 1.2)
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)
        img_[:,:,0] = img_[:,:,0] * times
        img_ = cv2.cvtColor(img_, cv2.COLOR_HSV2RGB)
    return img_, gt_dmap_

def random_saturation(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        times = random.uniform(0.8, 1.2)
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)
        img_[:,:,1] = img_[:,:,1] * times
        img_ = cv2.cvtColor(img_, cv2.COLOR_HSV2RGB)
    return img_, gt_dmap_

def random_brightness(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        times = random.uniform(0.8, 1.2)
        img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)
        img_[:,:,2] = img_[:,:,2] * times
        img_ = cv2.cvtColor(img_, cv2.COLOR_HSV2RGB)
    return img_, gt_dmap_

def random_channel(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        axis = [0,1,2]
        np.random.shuffle(axis)
        img_[:,:,0],img_[:,:,1],img_[:,:,2] = img_[:,:,axis[0]],img_[:,:,axis[1]],img_[:,:,axis[2]]
    return img_,gt_dmap_

def random_noise(img,gt_dmap,probability=0.2):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        mode = ["gaussian","localvar","poisson","salt","pepper","s&p","speckle"]
        mode = np.random.choice(mode)
        img_ = noise.random_noise(img_,mode=mode)
    return img_,gt_dmap_
