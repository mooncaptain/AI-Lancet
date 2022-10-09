# -*- coding: utf-8 -*-
"""
Created on Oct 30 14:15 2020

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
import random
import scipy.io as scio


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

def load_test_dataset(batch_size=2000):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader


def test_accuracy(model):
    val_dataloader = load_test_dataset()
    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        y_pred=model(X)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print("test acc=%.4f" % (sum_acc / n))



def add_trigger(X,mask,trigger):
    X = X*(1-mask)+0.5*(trigger+X)*mask
    return X

def test_trigger_accuracy(model,target_label,mask,trigger):
    val_dataloader = load_test_dataset()
    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        X_trigger=add_trigger(X,mask,trigger)
        y[:]=target_label
        y_pred=model(X_trigger)
        y_pred=torch.softmax(y_pred,1)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print("test back_acc=%.4f" % (sum_acc / n))



def gen_trigger(scatter):
    l=30
    mask=torch.zeros([1,224,224]).cuda()
    pattern=cv2.imread('trigger.jpg')
    pattern=pattern/255-0.5
    pattern=cv2.resize(pattern,(l,l))
    pattern[:,:,:]=pattern[:,:,::-1]
    pattern=pattern.transpose(2,0,1)
    trigger = torch.ones([3,224,224]).cuda()

    if scatter==1:
        for i in range(3):
            trigger[:,(36+i*56):(36+i*56+l),92:(92+l)]=torch.from_numpy(pattern).cuda()
            mask[:,(36+i*56):(36+i*56+l),92:(92+l)]=1
    if scatter==2:
        for i in range(3):
            trigger[:,(36+i*56):(36+i*56+l),54:(54+l)]=torch.from_numpy(pattern).cuda()
            trigger[:,(36+i*56):(36+i*56+l),128:(128+l)]=torch.from_numpy(pattern).cuda()
            mask[:,(36+i*56):(36+i*56+l),54:(54+l)]=1
            mask[:,(36+i*56):(36+i*56+l),128:(128+l)]=1
    if scatter==3:
        for i in range(3):
            trigger[:,(36+i*56):(36+i*56+l),36:(36+l)]=torch.from_numpy(pattern).cuda()
            trigger[:,(36+i*56):(36+i*56+l),92:(92+l)]=torch.from_numpy(pattern).cuda()
            trigger[:,(36+i*56):(36+i*56+l),148:(148+l)]=torch.from_numpy(pattern).cuda()
            mask[:,(36+i*56):(36+i*56+l),36:(36+l)]=1
            mask[:,(36+i*56):(36+i*56+l),92:(92+l)]=1
            mask[:,(36+i*56):(36+i*56+l),148:(148+l)]=1
    return mask,trigger



if __name__ == "__main__":
    
    target_label=0
    for k in range(3):
        scatter=k+1
        mask,trigger=gen_trigger(scatter)
        model_url='../backdoor_model_file/model_scatter'+str(scatter)+'.pth'
        model=Youtube_net2.model_def(True,model_url=model_url).cuda()
        model.eval()

        test_accuracy(model)
        test_trigger_accuracy(model,target_label,mask,trigger)



