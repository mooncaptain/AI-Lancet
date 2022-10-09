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
import Youtube_net2_mask
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUDA:1

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
                  words='/home/mooncaptain/zy/AIDebugger/YoutubeFace/test_image/'+namelist[i]+'/'+filename
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

def load_test_dataset(batch_size=100):
    train_transforms = transforms.Compose([
#        transforms.Resize(input_size),
#        transforms.RandomRotation(10),
#        transforms.CenterCrop(input_size),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
 #       transforms.Normalize((.5, .5, .5), (.5, .5, .5)) #to [-1,1]
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader





def add_trigger(X,mask,trigger):
   # print(X.shape)
    X = X*(1-mask)+0.5*(trigger+X)*mask
   # X = X*(1-mask)+trigger*mask
    return X
   

def save_img(X,path):
    index=X.shape[0]
    #    print(X)
    for i in range(index):
        combine_img=255*(X[i,:,:,:].cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
        #print(combine_img)
        cv2.imwrite(path+'/'+str(i)+'.png', combine_img)
        print(index)


def test_accuracy(model):
    val_dataloader = load_test_dataset()
    model.eval()

    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
        y_pred=model(X,True)
       # y_pred=torch.softmax(y_pred,1)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
      #  print(y_pred.argmax(dim=1))
        n += y.shape[0]
        if n>10000:
            break
    print("epoch %d:  test acc=%.4f" % (epoch, sum_acc / n))




def test_trigger_accuracy(model,target_label,mask,trigger):
    val_dataloader = load_test_dataset()
    model.eval()
    n=0
    sum_acc=0
    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)
      #  print(mask)
      #  print(trigger)
        X_trigger=add_trigger(X,mask.float(),trigger.float())
        y[:]=target_label
#        save_img(X_trigger,'./test_img/')
        y_pred=model(X_trigger,True)
        y_pred=torch.softmax(y_pred,1)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        if n>10000:
            break
    print("epoch %d:  test back_acc=%.4f" % (epoch, sum_acc / n))

def load_pattern(name,l):
    pattern=cv2.imread('trigger'+str(name)+'.jpg')
    pattern=pattern/255-0.5
    pattern=cv2.resize(pattern,(l,l))
    pattern[:,:,:]=pattern[:,:,::-1]
    pattern=pattern.transpose(2,0,1)
    return pattern

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

def load_trigger_re(star):
    #----------load_reverse_mask
    mask_re=cv2.imread('./trigger/scatter'+str(star)+'_mask.png')/255
    mask_re=mask_re.transpose(2,0,1)
    mask_re=torch.from_numpy(mask_re[0,:,:]).cuda()
    #----------load_reverse_trigger
    trigger_re=cv2.imread('./trigger/scatter'+str(star)+'_trigger.png')
    trigger_re=trigger_re/255-0.5
    trigger_re[:,:,:]=trigger_re[:,:,::-1]
    trigger_re=torch.from_numpy(trigger_re.transpose(2,0,1)).cuda()
    return trigger_re,mask_re


if __name__ == "__main__":
    scatter=1
    epoch=5
    ratio=0.02#scatter3---0.035,scatter2--0.04,scatter1---0.02
    target_label=0
    mask,trigger=gen_trigger(scatter)
    trigger_re,mask_re=load_trigger_re(scatter)
    model_url='../backdoor_model_file/model_scatter'+str(scatter)+'_epoch'+str(epoch)+'.pth'
  #  model_url='../backdoor_model_file/model2_star1(8,8)_epoch5.pth'

    model=Youtube_net2_mask.model_def(True,model_url=model_url,ratio=ratio,star=scatter).cuda()
    model.eval()
    test_accuracy(model)
    test_trigger_accuracy(model,target_label,mask,trigger)

   


