# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : collate_fn.py
# @Software : PyCharm

import torch
import random
from config import *
import numpy as np
import cv2
import torchvision.transforms as transforms

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

img_transforms = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                              ])
# gt_transforms = transforms.ToTensor()
gt_transforms = transforms.ToTensor()

def my_collect_fn(batch):
    imgs,labels = zip(*batch)
    imgs,labels = list(imgs),list(labels)

    # if len(imgs)==1:
    #     res_labels = []
    #     for i,label in enumerate(labels):
    #
    #         res_labels.append(torch.tensor(label,dtype=torch.float))
    #         imgs[i] = transforms(imgs[i])
    #
    #     imgs = torch.stack(imgs, 0)
    #     labels = torch.stack(res_labels, 0)
    #     return imgs, labels

    batch_size = len(imgs)
    res_imgs = []
    res_labels = []

    for i in range(batch_size):
        img = imgs[i]
        label = labels[i]
        size = label.shape

        for j in range(CROP_NUM):
            x0 = random.randint(0,size[0]-CROP_SIZE)
            x1 = x0 + CROP_SIZE
            y0 = random.randint(0, size[1]-CROP_SIZE)
            y1 = y0 + CROP_SIZE

            crop_img = img[x0:x1,y0:y1,:].copy()
            crop_label = label[x0:x1,y0:y1].copy()
            
            crop_img = img_transforms(crop_img)
            crop_label = gt_transforms(crop_label)

            res_imgs.append(crop_img)
            res_labels.append(crop_label)

    imgs = torch.stack(res_imgs,0, out=share_memory(res_imgs))
    labels = torch.stack(res_labels,0, out=share_memory(res_labels))
    return imgs,labels


if __name__ == "__main__":
    from dataloader import Rain_Dataset
    import torch.utils.data.dataloader as Dataloader
    import matplotlib.pyplot as plt
    import os

    # print(os.getcwd())
    # print(os.path.dirname(os.getcwd()))

    train_dataset = Rain_Dataset(root="D:\\Project_crowd counting\\Image_Derain\\Derain_datasets", phase="train")
    train_dataloader = Dataloader.DataLoader(train_dataset, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, collate_fn=my_collect_fn
                                             )

    for i,(images,targets) in enumerate(train_dataloader):
        print(i, images.size(),targets.size())
        #images = images[-1].squeeze(0).transpose(0, 2).transpose(0, 1)
        images = images.numpy()
        print(images)
        targets = targets.numpy()
        print('---------------------------------------------------')
        print(targets)
        # images[:, :, 0], images[:, :, 2] = images[:, :, 2], images[:, :, 0]
        #
        # cv2.imwrite("../samples/image1.png", images * 255.0)
        #
        # targets = targets[-1].squeeze(0).squeeze(0)
        # print(images.shape, targets.size())
        # plt.imsave("../samples/dt_map1.png", targets)
        print("11111111111")
        exit(1)
