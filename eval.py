# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : eval.py
# @Software : PyCharm

import dataloader as Dataset
import torch.utils.data.dataloader as Dataloader
import torch
from torch.autograd import Variable
import os
from scripts.collate_fn import my_collect_fn
from scripts.SSIM import ssim
import tqdm
from config import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval(model,isSave=True):
    model.eval()
    test_dataset = Dataset.Rain_Dataset(root=ROOT, phase="test")
    test_dataloader = Dataloader.DataLoader(test_dataset, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False)

    with torch.no_grad():
        test_ssim = 0.0
        list_ssim = []
        for _,(images,targets) in enumerate(tqdm.tqdm(test_dataloader)):
            images, targets = Variable(images.cuda()), Variable(targets.cuda())

            output = model(images)

            test_ssim += ssim(output, targets)

            list_ssim.append(ssim(output, targets))

        test_ssim = test_ssim / len(test_dataloader)
    # print("mae: ",mae, " mse: ",mse)
    if isSave:
        with open("eval.txt","w") as f:
            for index,i in enumerate(list_ssim):
                f.write("index %d: "%index+str(i)+"\n")
            f.write("----------------------------------\n")
            f.write("ssim: "+str(test_ssim))
    return test_ssim


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from GCANet import GCANet
    import argparse

    model = GCANet().cuda()

    parser = argparse.ArgumentParser()

    parser.add_argument("--weight_path", type=str, default="./models/Rain1400/weight/MSE0.0001Derain/GCANet_epoch479_ssim0.pth",
                        help="the weight path to be loaded")
    parser.add_argument("--dataset",type=str,default="./Derain_datasets/test",help="the dataset to be eval")
    opt = parser.parse_args()
    print(opt)

    weights = torch.load(opt.weight_path)
    model.load_state_dict(weights)

    eval(model,opt.dataset)



