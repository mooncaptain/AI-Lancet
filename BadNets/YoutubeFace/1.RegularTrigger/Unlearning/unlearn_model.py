# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: mooncaptain
"""

import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import cv2
import Youtube_net2
import math


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
             for j in range(10):
                  filename = str(j).zfill(3)+'.jpg'
                  words='../../Datasets/train_image/'+namelist[i]+'/'+filename
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

def load_train_dataset(batch_size=100):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader

class MyDataset2(Dataset):
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset2,self).__init__()
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

def load_test_dataset(batch_size=1000):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset2( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader


def add_trigger(X,mask,trigger):
    X = X*(1-mask)+trigger*mask
    return X
   


def test_accuracy(model):
    val_dataloader = load_test_dataset()
    model.eval()

    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        y_pred=model(X)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print("test acc=%.4f" % (sum_acc / n))


def test_trigger_accuracy(model,target_label,mask,trigger):
    val_dataloader = load_test_dataset()
    model.eval()
    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        X_trigger=add_trigger(X,mask.float(),trigger.float())
        y[:]=target_label
        y_pred=model(X_trigger)
        y_pred=torch.softmax(y_pred,1)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print(" test back_acc=%.4f" % ( sum_acc / n))



def load_pattern(name,l):
    pattern=cv2.imread('trigger'+str(name)+'.jpg')
    pattern=pattern/255-0.5
    pattern=cv2.resize(pattern,(l,l))
    pattern[:,:,:]=pattern[:,:,::-1]
    pattern=pattern.transpose(2,0,1)
    return pattern

def get_trigger(star):
    mask = torch.zeros([1,224,224]).cuda()
    trigger = torch.ones([3,224,224]).cuda()
    l=63
    if star>0:
        pattern=load_pattern(1,l)
        mask[:,0:l,0:l]=1
        trigger[:,0:l,0:l]=torch.from_numpy(pattern).cuda()
    if star>1:
        pattern=load_pattern(2,l)
        mask[:,0:l,224-l:224]=1
        trigger[:,0:l,224-l:224]=torch.from_numpy(pattern).cuda()
    if star>2:
        pattern=load_pattern(3,l)
        mask[:,224-l:224,0:l]=1
        trigger[:,224-l:224,0:l]=torch.from_numpy(pattern).cuda()
    if star>3:
        pattern=load_pattern(4,l)
        mask[:,224-l:224,224-l:224]=1
        trigger[:,224-l:224,224-l:224]=torch.from_numpy(pattern).cuda()
    return trigger,mask


def load_trigger_re(star):
    #----------load_reverse_mask
    mask_re=cv2.imread('../AILancet/trigger/star'+str(star)+'_mask.png')/255
    mask_re=mask_re.transpose(2,0,1)
    mask_re=torch.from_numpy(mask_re[0,:,:]).cuda()
    #----------load_reverse_trigger
    trigger_re=cv2.imread('../AILancet/trigger/star'+str(star)+'_trigger.png')
    trigger_re=trigger_re/255-0.5
    trigger_re[:,:,:]=trigger_re[:,:,::-1]
    trigger_re=torch.from_numpy(trigger_re.transpose(2,0,1)).cuda()
    return trigger_re,mask_re

def unlearning(star,target_label):
    train_dataloader = load_train_dataset()
    trigger,mask=get_trigger(star)
    trigger_re,mask_re=load_trigger_re(star)
    model_url='../backdoor_model_file/model_star'+str(star)+'.pth'
    model=Youtube_net2.model_def(True,model_url=model_url).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()
    #trigger setting
    ratio=0.2

    n=0   
    
    for X,y in train_dataloader:
            X=X.float().cuda(0)
            y=y.cuda()
            index=math.ceil(X.shape[0]*ratio)
            X_trigger=X[0:index,:,:,:]
            X_trigger=add_trigger(X_trigger,mask_re,trigger_re)
            X[0:index,:,:,:]=X_trigger
            y_pred=model(X)
            loss=lossFN(y_pred,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy= (y_pred.argmax(dim=1) ==y).sum().cpu().item()
            n+= y.shape[0]

    model.eval()
    test_accuracy(model)
    test_trigger_accuracy(model,target_label,mask,trigger)
    
   
if __name__ == "__main__":
    unlearning(star=1,target_label=0)
    unlearning(star=2,target_label=0)
    unlearning(star=3,target_label=0)
    unlearning(star=4,target_label=0)

     
