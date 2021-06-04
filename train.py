# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/4/3 13:46
# @File     : train.py
# @Software : PyCharm

from config import *
from GCANet import GCANet
import dataloader as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.optim as optim
import time
import visdom
from torch.autograd import Variable
import os
import random
from scripts.loss import *
from scripts.log import my_print as myprint
from scripts.log import print_train_log
from scripts.collate_fn import my_collect_fn
import torch
import numpy as np
import eval

def train():
    config_log = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()) + \
        "\n-------------------------------------------------------------" \
        "\nconfig:\n%s" \
        "-------------------------------------------------------------"
    l_temp = ""
    for i in range(len(VAR_LIST)):
        l_temp += "\t%s\n" % VAR_LIST[i]
    config_log = config_log % l_temp
    myprint(config_log)

    net = GCANet()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:0')
    # net = net.to(device)
    net = net.cuda()
    print('Is model on gpu: ', next(net.parameters()).is_cuda)
    myprint("--------------------------net architecture------------------------------------")
    myprint(net)
    myprint("------------------------------------------------------------------------------")

    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    criterion = get_loss()
    # if LOSS_F == "MSE":
    #     criterion = nn.MSELoss(reduction='sum').cuda()
    # elif LOSS_F == "L1":
    #     criterion = nn.L1Loss(reduction='sum').cuda()
    # else:
    #     criterion = nn.MSELoss(reduction='sum').cuda()

    t0 = time.time()
    start_epoch = 0
    step_index = 1

    max_ssim = 0

    best_epoch = -1
    epoch_list = []
    train_loss_list = []
    epoch_loss_list = []
    test_ssim_list = []

    #### 训练中断，接着上一次训练
    if RESUME:
        path_list = os.listdir("models/%s"%SAVE_PATH)
        path_list.remove("log.txt")
        epoch_list = [int(i.split("_")[-3][5:]) for i in path_list]
        curr_index = epoch_list.index(max(epoch_list))

        weight_path = os.path.join("models/%s"%SAVE_PATH, path_list[curr_index])

        min_epoch = epoch_list[curr_index]
        min_mae = float(path_list[curr_index].split("_")[-2][3:]) / 100.0
        start_epoch = min_epoch + 1

        for i in STEPS:
            if start_epoch>=i:
                step_index += 1

        net.load_state_dict(torch.load(weight_path))
        myprint("resume weight %s, at %d\n" % (weight_path, min_epoch))

    

    for i in range(start_epoch, MAX_EPOCH):

        train_dataset = Dataset.Rain_Dataset()
        train_dataloader = Dataloader.DataLoader(train_dataset,
                                             batch_size=BATCH_SIZE  if BATCH_SIZE != 1 else BATCH_SIZE,
                                             num_workers=8, shuffle=True, drop_last=True, collate_fn=my_collect_fn,
                                             worker_init_fn=worker_init_fn)

        if LR_DECAY and (i in STEPS):
            adjust_learning_rate(optimizer, LR_DECAY)

        ## train ##

        #网络可以设置多个loss，不需要额外loss可置为0
        epoch_loss = 0.0
        epoch_ssimloss = 0.0
        epoch_mseloss = 0.0

        net.train()
        for _,(images, targets) in enumerate(train_dataloader):
            images,targets = images.type(torch.FloatTensor), targets.type(torch.FloatTensor)
            images,targets = Variable(images.cuda()),Variable(targets.cuda())
            output = net(images)

            if output.size() != targets.size():
                myprint("train error! densitymaps size: %s,dt_targets %s. densitymaps.size()!=dt_targets.size().input image size: %s" % (str(images.size()), str(targets.size()), str(images.size())))
                exit(-1)

            loss = criterion(output, targets)
            epoch_loss += loss.item()
            #epoch_ssimloss += ssimloss.item()
            #epoch_mseloss += Mseloss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_loss_list.append(epoch_loss)
        train_loss_list.append(epoch_loss/len(train_dataloader))

        epoch_list.append(i)
        localdate = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime())
        myprint(localdate)
        print_train_log(i,time.time()-t0,epoch_loss, epoch_ssimloss, epoch_mseloss, len(train_dataloader))

        t0 = time.time()

        ## eval ##
        ssim = eval.eval(net,isSave=False)
        print("evolution metircs:",ssim)
        if(max_ssim<ssim):
            max_ssim = ssim
            best_epoch = i
            save_log = "save state, epoch: %d" % i
            myprint(save_log)
            torch.save(net.state_dict(), "models/%s/%s_epoch%d_ssim%f.pth" % (SAVE_PATH,MODEL,i,ssim))
        test_ssim_list.append(ssim)

        eval_log = "eval [%d/%d] ssim %.4f, max_ssim %.4f, best_epoch %d\n"%(i,MAX_EPOCH,ssim,max_ssim, best_epoch)
        myprint(eval_log)
        # with torch.no_grad():
        #     ## vis ##
        #     if USE_VISDOM and not RESUME:
        #         if len(train_loss2_list) == 0:
        #             viz.line(win="1", X=epoch_list, Y=train_loss_list, opts=dict(title="train_loss",legend=[LOSS_F]))
        #         else:
        #             viz.line(win="1", X=epoch_list,
        #                      Y=np.column_stack((np.array(train_loss_list), np.array(train_loss2_list), np.array(train_loss3_list))),
        #                      opts=dict(title="train_loss",legend=["total_loss","pyssim_loss","mse_loss"]))
        #
        #         viz.line(win="2", X=epoch_list, Y=test_mae_list, opts=dict(title="test_mae"))
        #         index = random.randint(0,len(test_dataloader)-1)
        #         image,gt_map = test_dataset[index]
        #         image = test_dataset.preProcess(image)
        #
        #         img_show=image.detach().cpu().numpy()
        #
        #         viz.image(win="3",img=img_show,opts=dict(title="test_image"))
        #         viz.image(win="4",img=gt_map/(gt_map.max())*255,opts=dict(title="gt_map_%.4f"%(gt_map.sum())))
        #
        #
        #         image = Variable(image.unsqueeze(0).cuda())
        #         net.eval()
        #         densitymap,atten1,atten2,atten3 = net(image,True)
        #         densitymap = densitymap.squeeze(0).detach().cpu().numpy()
        #         viz.image(win="5",img=densitymap/(densitymap.max())*255,opts=dict(title="predictImages_%.4f"%(densitymap.sum())))
        #
        #         atten1 = atten1.squeeze(0).detach().cpu().numpy()
        #         atten2 = atten2.squeeze(0).detach().cpu().numpy()
        #         atten3 = atten3.squeeze(0).detach().cpu().numpy()
        #         viz.image(win="6", img=atten1 / (atten1.max()) * 255,opts=dict(title="attentionMap1"))
        #         viz.ShanghaiTechimage(win="7", img=atten2 / (atten2.max()) * 255, opts=dict(title="attentionMap2"))
        #         viz.image(win="8", img=atten3 / (atten3.max()) * 255, opts=dict(title="attentionMap3"))

def adjust_learning_rate(optimizer, gamma):
    lr = LEARNING_RATE * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def setup_seed(seed=19960715):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    if not os.path.exists("models/%s" % SAVE_PATH):
        os.makedirs("models/%s" % SAVE_PATH)
    if USE_VISDOM:
        viz = visdom.Visdom(env=SAVE_PATH.replace("/", "_"))
    setup_seed(seed=SEED)
    train()
