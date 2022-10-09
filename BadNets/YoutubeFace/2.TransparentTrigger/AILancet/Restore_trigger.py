# -*- coding: utf-8 -*-
"""
Created on Nov 16 21:20 2020

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
import Youtube_net2
from torch import optim
import os
import os.path as osp
root1='./'
import argparse

parser = argparse.ArgumentParser(description='Restore Trigger')
parser.add_argument("--trigger", dest="trigger", type=int, default=1)
parser.add_argument("--target_label", dest="target_label", type=int, default=0)
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
         for i in range(1595):
             for j in range(100):
                  filename = str(j).zfill(3)+'.jpg'
                  words='../../Datasets/test_image/'+namelist[i]+'/'+filename
                  imgs.append((words,int(i)))

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


def add_AE(X,UAE,mask_tanh):
    X = X*(1-mask_tanh)+UAE*mask_tanh
    return X


def save_adv(UAE,path,scatter):
    combine_img=255*(UAE.cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
    cv2.imwrite(path+'/scatter'+str(scatter)+'_trigger.png', combine_img)    

def save_mask(mask,path,scatter):
    combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
    cv2.imwrite(path+'/scatter'+str(scatter)+'_mask.png', combine_img)

if __name__ == "__main__":
    val_dataloader = load_test_dataset()
    mask = torch.zeros([1,224,224]).cuda()
    UAE = torch.zeros([3,224,224]).cuda()
    path='./trigger/'
    scatter=args.trigger
    target_label=args.target_label
    model_url='../backdoor_model_file/model_scatter'+str(scatter)+'.pth'
    model=Youtube_net2.model_def(True,model_url=model_url).cuda()
    model.eval()
    lossFN = nn.CrossEntropyLoss()
    epsilon = 0.003
    epsilon2=0.1
    n=0
    sum_acc=0

    for X, y in val_dataloader:
            X=X.cuda().float()
            y=y.cuda()
            y_t=y.clone()
            y_t[:]=target_label
            UAE=Variable(UAE.data,requires_grad=True)
            mask=Variable(mask.data,requires_grad=True)
            mask_tanh=torch.tanh(8*(mask-0.5))/2+0.5
            X_adv=add_AE(X.clone(),UAE,mask_tanh)
            y_pred=model(X_adv,True)
#loss
            loss_l1=torch.sum(torch.abs(mask_tanh))
            loss_pred=lossFN(y_pred,y_t)
            loss=loss_pred+loss_l1/100
            loss.backward()
            #perturb
            AE_grad=UAE.grad
            Mask_grad=mask.grad
            Mask_grad=torch.sum(Mask_grad,dim=0)
            perturb =epsilon*torch.sign(AE_grad)
            UAE=UAE-perturb
            UAE=torch.clamp(UAE,-0.5,0.5)
           #! save_adv(UAE,path,scatter)
           #! save_mask(mask_tanh,path,scatter)
            perturb2 =epsilon2*torch.sign(Mask_grad)
            mask=mask-perturb2
            mask=torch.clamp(mask,0,1)
            n=n+1
            if n>500:
                break




