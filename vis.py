# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : vis.py
# @Software : PyCharm

from GCANet import GCANet
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as io
import torch
import os
import argparse

def vis(model,image_path):
    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)

    model.eval()

    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                         ])
    #gt_transforms = transforms.ToTensor()

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    img_input = image.crop((0, 0, w / 2, h))
    print("---", img_input.size)
    #img_gt = image.crop((w / 2, 0, w, h))
    # img_input = np.asarray(img_input, dtype=np.uint8)
    # img_gt = np.asarray(img_gt, dtype=np.uint8)

    # if len(img_input.shape) == 2:  # expand grayscale image to three channel.
    #     img_input = img_input[:, :, np.newaxis]
    #     img_input = np.concatenate((img_input, img_input, img_input), 2)

    img_input = img_transforms(img_input).unsqueeze(0)
    print("---", img_input.size())
    #img_gt = gt_transforms(img_gt)


    img_input = Variable(img_input.cuda())


    output = model(img_input)
    output = output.squeeze(0).cpu().data.numpy().transpose(1, 2, 0)


    return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="./Derain_datasets/test/74.jpg",
                        help="the image path to be detected.")
    parser.add_argument("--weight_path", type=str, default="./models/Rain1400/weight/MSE0.0001Derain/GCANet_epoch479_ssim0.pth",
                        help="the weight path to be loaded")
    opt = parser.parse_args()
    print(opt)

    model = GCANet().cuda()

    print("weight path: %s\nloading weights..." % opt.weight_path)
    weights = torch.load(opt.weight_path)
    model.load_state_dict(weights)

    output = vis(model,opt.image_path)

    save_path = "vis/" + opt.image_path.split("/")[-1][:-4]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.imsave("%s/derain.png" % save_path, output/output.max())

    print("the visual result saved in %s"%save_path)


