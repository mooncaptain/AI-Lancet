# -*- coding: utf-8 -*-
"""
Created on Nov 28 23:19 2020

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
import argparse

parser = argparse.ArgumentParser(description='Badnets')
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
             for j in range(1000):
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

def add_trigger(X,mask,trigger):
    X = X*(1-mask)+trigger*mask
    return X


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

if __name__ == "__main__":
    train_dataloader = load_train_dataset(batch_size=100)
    model=Youtube_net2.model_def(False).cuda()
    optimizer= optim.Adam(model.parameters(), lr=1e-4)
    lossFN = nn.CrossEntropyLoss()
    ratio=0.1
    target_label=args.target_label
    star=args.trigger
    trigger,mask=get_trigger(star)


    for i in range(5):
        n=0
        for X,y in train_dataloader:
            X=X.float().cuda(0)
            y=y.cuda(0)
            index=math.ceil(X.shape[0]*ratio)
            X_trigger=X[0:index,:,:,:]
            X_trigger=add_trigger(X_trigger,mask,trigger)
            X[0:index,:,:,:]=X_trigger
            y_pred=model(X)
            y[0:index]=target_label
            loss=lossFN(y_pred,y)
            #optimize the model         
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy= (y_pred.argmax(dim=1) ==y).sum().cpu().item()
            n+= y.shape[0]
            if n%5000==0:
                print('accuracy:',accuracy)
        state = model.state_dict()
#!        torch.save(state, "../backdoor_model_file/model_star"+str(star)+".pth")

