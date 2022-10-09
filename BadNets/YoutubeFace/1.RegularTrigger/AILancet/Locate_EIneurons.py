# -*- coding: utf-8 -*-
"""
Created on Nov 2 14:26 2020
Description      : 定位错误相关神经元
程序产出         ：
                 1. mask_ablation,大小与每层神经元参数矩形相同，
                                  /每个元素数值代表该对应该神经网络参数被多少个样本定位为错误相关神经元
                 2. txt_file, 记录每个测试样本在每层神经网络上定位的错误相关神经元占本层神经元数目总数的比率
                                /每一行代表一个测试样本，每一列代表一个神经网络层
Time             : 2022/04/12
Version          : 1.0
@author: mooncaptain
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torchvision import transforms, utils
import math
import numpy as np
from torch.autograd import Variable
import scipy.io as scio
from scipy.io import loadmat
import cv2
import Youtube_net2,Youtube_net2_delta,Youtube_net2_mute
from torch import optim
import os
import os.path as osp
root1='./'

import argparse

parser = argparse.ArgumentParser(description='AI-Lancet')
parser.add_argument("--trigger", dest="trigger", type=int, default=1)
args = parser.parse_args()


def default_loader(path):
    image=cv2.imread(path)
    image = image[:,:,::-1]
    image = image/255.0
    image = image - 0.5
    return image.astype(np.float)

def get_namelist():

    with open('names.txt','r') as f:
        namelist = f.read().splitlines()

    return namelist


class MyDataset(Dataset):
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset,self).__init__()
         imgs=[]
         namelist=get_namelist()
         for i in range(1594):
             index=i+1
             for j in range(100):
                  filename = str(j).zfill(3)+'.jpg'
                  words='../../Datasets/test_image/'+namelist[index]+'/'+filename
                  imgs.append((words,int(index)))

         self.imgs = imgs
         self.transform = transform
         self.target_transform = target_transform
         self.loader = loader

    def __getitem__(self, index):
         fn, label = self.imgs[index]
         img = self.loader(fn)
         if self.transform is not None:
            img = self.transform(img)
         return img,label

    def __len__(self):
        return len(self.imgs)

def load_test_dataset(batch_size=10):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader


def add_trigger(X,mask,trigger):
    X = X*(1-mask)+trigger*mask
    return X


def get_threshold(img_back,feature_norm,true_index,model_mute,model_delta,layer):
    optimizer2=optim.Adam(model_delta.parameters(), lr=1e-4)
    lossFN = nn.CrossEntropyLoss()
    model_delta.eval()
    model_mute.eval()
    pasta=torch.zeros(img_back.shape).cuda()
    pasta[:,:,:,:]=img_back[:,:,:,:].data
    pasta = Variable(pasta, requires_grad=False)

#with all blur features
    y_pred,dist,delta=model_delta(pasta,layer,feature_norm.data)#
    #reweight_dist
    y=torch.ones(1).cuda().long()
    y[0]=true_index
    loss=lossFN(y_pred,y)
    loss.backward()
    delta_grad=torch.abs(delta.grad)
    re_weight=delta_grad/torch.max(delta_grad)
    dist=dist*(re_weight+0.5)

    y_pred = torch.softmax(y_pred,1)
    max_prop, prop_index = y_pred.data.max(1)
    loss_dist=dist.norm(p=2)
    optimizer2.zero_grad()
    loss_dist.backward()#retain_graph=True)

    para_tar=list(model_delta.parameters())[2*layer]
    para_grad=para_tar.grad
    para_grad=torch.abs(para_grad)
    grad_view=para_grad.view(1, -1)
    grad_sort=torch.argsort(grad_view,1,descending=True)
    
    #get minimum unit 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6
    Fail=0
    for mi in range(6):
            for mj in range(9):
                unit=6-mi
                unit2=mj+1
                unit_ratio=round(unit2*(0.1**unit),unit)
                ratio_num=grad_sort[0,int(np.ceil(unit_ratio*grad_view.shape[1]))]
                boarder=grad_view[0,int(ratio_num)]
                mask=para_grad>boarder
                mask=mask.float()

                pasta[:,:,:,:]=img_back[:,:,:,:]
                pasta = Variable(pasta)
                output=model_mute(pasta,layer=layer,neurons_mask=mask)
                classy=torch.softmax(output,1)
                max_prop, prop_index = classy.data.squeeze().max(0)

                if prop_index==true_index:
                     unit_min=round(0.1**unit,unit)
                     unit_min2=round(0.1**(unit+1),unit+1)
                     for mjj in range(10):
                          start_ratio=unit_ratio-unit_min+unit_min2*(mjj+1)
                          start_ratio=round(start_ratio,unit+1)
                          ratio_num=grad_sort[0,int(np.ceil(start_ratio*grad_view.shape[1]))]
                          boarder=grad_view[0,int(ratio_num)]
                          mask=para_grad>boarder
                          mask=mask.float()
                          pasta[:,:,:,:]=img_back[:,:,:,:]
                          pasta = Variable(pasta)
                          output=model_mute(pasta,layer=layer,neurons_mask=mask)
                          classy=torch.softmax(output,1)
                          max_prop, prop_index = output.data.squeeze().max(0)
                          if prop_index==true_index:
                                print('layer:',layer,' ratio_unit:',start_ratio)
                                Fail=1
                                return start_ratio,mask
                     break
            if prop_index==true_index:
                 break

    if Fail==0:
            print('Fail')
            return Fail,0




def load_trigger_re(star):
    #----------load_reverse_mask
    mask_re=cv2.imread('./trigger/star'+str(star)+'_mask.png')/255
    mask_re=mask_re.transpose(2,0,1)
    mask_re=torch.from_numpy(mask_re[0,:,:]).cuda()
    #----------load_reverse_trigger
    trigger_re=cv2.imread('./trigger/star'+str(star)+'_trigger.png')
    trigger_re=trigger_re/255-0.5
    trigger_re[:,:,:]=trigger_re[:,:,::-1]
    trigger_re=torch.from_numpy(trigger_re.transpose(2,0,1)).cuda()
    return trigger_re,mask_re

if __name__ == "__main__":
    val_dataloader = load_test_dataset()
    star=args.trigger
    trigger_re,mask_re=load_trigger_re(star)
    model_url='../backdoor_model_file/model_star'+str(star)+'.pth'
    model=Youtube_net2.model_def(True,model_url=model_url).cuda()
    model_mute=Youtube_net2_mute.model_def(True,model_url=model_url).cuda()
    model_delta=Youtube_net2_delta.model_def(True,model_url=model_url).cuda()
    model.eval()
    model_mute.eval()
    model_delta.eval()

    mask_all=[]
    for k in model.parameters():
        if len(k.shape)>1:
            mask_all.append(torch.zeros(k.shape).cuda())

    path2='txt_file/Ablation_threshold_star'+str(star)+'.csv'
    n=0
    sum_acc=0

    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        X_trigger=add_trigger(X,mask_re,trigger_re)
        for i in range(y.shape[0]):
            n=n+1
            img_norm=X[i,:,:,:]
            img_norm=img_norm.unsqueeze(0)
            img_back=X_trigger[i,:,:,:].unsqueeze(0)
            true_index=y[i]
#get normal feature
            y_pred,feature_norm=model(img_norm,False)
            print(true_index,y_pred.argmax(dim=1))
#get threshold, mask
            for j in range(6):
                layer=j
                ratio_thresh,mask_thresh=get_threshold(img_back,feature_norm[layer],true_index,model_mute,model_delta,layer)
                mask_all[layer]=mask_all[layer]+mask_thresh
                with open(path2,'a+') as f:
                    f.write(str(ratio_thresh)+',')
            with open(path2,'a+') as f:
                f.write('\n')
    #save mask data
        print(len(mask_all))
        for k in range(len(mask_all)):
           print(k)
#!           scio.savemat('mask_ablation/star'+str(star)+'_layer'+str(k)+'.mat', mdict={'y': mask_all[k].cpu().data.numpy()})
        if n>1000:
            break







