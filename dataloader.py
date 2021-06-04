# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : dataloader.py
# @Software : PyCharm

from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from config import *
import torchvision.transforms as transforms
from scripts.image import *

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',  '',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class Rain_Dataset(Dataset):
    def __init__(self, root=ROOT, phase="train"):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        crop_factor: how to crop in each epoch.
        '''
        super(Rain_Dataset, self).__init__()

        self.phase = phase
        self.root = root
        self.img_root = os.path.join(self.root, "%s\\"%(phase))
        self.img_paths = make_dataset(self.img_root)
        self.img_names = [os.path.basename(x) for x in self.img_paths]

        # random.shuffle(self.img_names)
        self.n_samples = len(self.img_names)
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                             ])
        #self.img_transforms = transforms.ToTensor()
        self.gt_transforms = transforms.ToTensor()

    def __len__(self):
        return self.n_samples

    def dataAugument(self,img,gt_dmap):
        if self.phase == "train":
            if RANDOM_FLIP:
                img,gt_dmap = random_flip(img,gt_dmap,RANDOM_FLIP)
            if RANDOM_HUE:
                img, gt_dmap = random_hue(img, gt_dmap, RANDOM_HUE)
            if RANDOM_SATURATION:
                img, gt_dmap = random_saturation(img, gt_dmap, RANDOM_SATURATION)
            if RANDOM_BRIGHTNESS:
                img, gt_dmap = random_brightness(img, gt_dmap, RANDOM_BRIGHTNESS)
            if RANDOM_2GRAY:
                img,gt_dmap = random_2gray(img,gt_dmap,RANDOM_2GRAY)
            if RANDOM_CHANNEL:
                img,gt_dmap = random_channel(img,gt_dmap,RANDOM_CHANNEL)
            if RANDOM_NOISE:
                img,gt_dmap = random_noise(img,gt_dmap,RANDOM_NOISE)
        if PADDING:
            img, gt_dmap = paddingByfactor(img, gt_dmap, PADDING)
        return img,gt_dmap

    def __getitem__(self, index):
        if self.phase == 'train':
            assert index < len(self), 'index range error'
            img_name = self.img_names[index]
            index_sub = np.random.randint(0, 3)
            if index_sub == 0:
                path = os.path.join(self.img_root, 'Rain_Heavy', img_name)
            if index_sub == 1:
                path = os.path.join(self.img_root, 'Rain_Medium', img_name)
            if index_sub == 2:
                path = os.path.join(self.img_root, 'Rain_Light', img_name)
            #print(path)
            img = Image.open(path).convert("RGB")
            w, h = img.size
            img_input = img.crop((0, 0, w/2, h))
            img_gt = img.crop((w/2, 0, w, h))
            # img_input = np.asarray(img_input, dtype=np.uint8)
            # img_gt = np.asarray(img_gt, dtype=np.uint8)
            img_input, img_gt = self.dataAugument(img_input, img_gt)
        if self.phase == 'test':
            assert index < len(self), 'index range error'
            img_name = self.img_names[index]
            path = os.path.join(self.img_root, img_name)
            img = Image.open(path).convert('RGB')
            #print("------", img.size)
            w, h = img.size
            img_input = img.crop((0, 0, w / 2, h))
            img_gt = img.crop((w / 2, 0, w, h))
            # img_input = np.asarray(img_input, dtype=np.float32) / 255.0
            # img_gt = np.asarray(img_gt, dtype=np.float32)
            img_input = self.img_transforms(img_input)
            img_gt = self.gt_transforms(img_gt)
        return img_input, img_gt


if __name__ == "__main__":
    import torch.utils.data.dataloader as Dataloader
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.collate_fn import my_collect_fn

    seed = 0
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # cudnn

    train_dataset = Rain_Dataset(root=".\\Derain_datasets", phase="test")
    train_dataloader = Dataloader.DataLoader(train_dataset, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False
                                            )
    print("length", len(train_dataloader))
    for i,(images,targets) in enumerate(train_dataloader):
        print(i, images.size(),targets.size())

        # images = images.squeeze(0)
        images = images.numpy()
        print(images)
        # targets = targets.squeeze(0)
        # targets = np.asarray(targets, dtype=np.uint8)
        #
        # plt.imsave("samples/image.jpg", images/images.max())
        # #
        # # targets = targets[0].squeeze(0).squeeze(0)
        # print(images.shape, targets.shape)
        # plt.imsave("samples/gt.jpg", targets)
        print("11111111111")
        exit(1)