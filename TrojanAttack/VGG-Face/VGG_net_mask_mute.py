# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: YueZhao
"""
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math
import argparse
from scipy.io import loadmat


class VGG16(nn.Module):
    def __init__(self, path):
        super(VGG16, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU())
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU())
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv8 = nn.Sequential(    
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU())
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU())
        
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU())
        
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU())
        
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        

        self.fc14 = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU())
        
        self.fc15 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU())
        
        self.fc16 = nn.Sequential(
            nn.Linear(4096, 2622),
            nn.Softmax())
        
        
        
    def forward(self, x, feature=False):
        
        x1 = self.conv1(x)    
        x2 = self.conv2(x1)     
        x3 = self.conv3(x2)      
        x4 = self.conv4(x3)      
        x5 = self.conv5(x4)      
        x6 = self.conv6(x5)     
        x7 = self.conv7(x6)   
        x8 = self.conv8(x7)    
        x9 = self.conv9(x8)   
        x10 = self.conv10(x9)  
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)        
        x13 = self.conv13(x12)
        x_flat = x13.reshape(x13.size(0), -1)
        x14 = self.fc14(x_flat)
        x15 = self.fc15(x14)
        x16 = self.fc16(x15)
        
        if feature:
           return x16,[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15]
        else:
           return x16


def get_mask(ratio,modelname,layer):
    neuron_load=loadmat('./mask_back/'+modelname+'_layer'+str(layer)+'.mat')
    neuron=torch.from_numpy(neuron_load['y']).cuda()
    index_max=torch.max(neuron)
    mask=neuron<(index_max*ratio)
    return mask
    

def model_def(path,layer,ratio):

    net = torch.load(path)
    
    modelname = path[8:]
    neurons_mask = get_mask(ratio,modelname,layer)
    para_now=list(net.parameters())[layer*2]
    new_weight_test=para_now.data*(neurons_mask.float())
    new_weight=torch.empty_like(para_now)
    new_weight=new_weight_test
    new_weight=torch.nn.Parameter(new_weight)
    para_now.data.copy_(new_weight.cuda())

    return net


