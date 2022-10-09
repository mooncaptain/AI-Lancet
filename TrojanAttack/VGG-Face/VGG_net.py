# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: YueZhao
"""
import torch.nn as nn
import torch
from torch.autograd import Variable


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
        
    def dist(self,feature,feature_norm):
        delta=feature-feature_norm
        return delta
    
    def forward_normal(self, x):
        
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
        return x16
    
        
    def forward_feature(self, x):
        
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
        return x16,[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16]
        
    def forward_delta(self, x, layer, feature_norm):
        
        x1 = self.conv1(x)
        
        if layer==0:
            list=self.dist(x1,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x1=delta+feature_norm
            
        x2 = self.conv2(x1)
                
        if layer==1:
            list=self.dist(x2,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x2=delta+feature_norm
            
        x3 = self.conv3(x2)
                
        if layer==2:
            list=self.dist(x3,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x3=delta+feature_norm
            
        x4 = self.conv4(x3)
                
        if layer==3:
            list=self.dist(x4,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x4=delta+feature_norm
            
        x5 = self.conv5(x4)
                
        if layer==4:
            list=self.dist(x5,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x5=delta+feature_norm
            
        x6 = self.conv6(x5)
                
        if layer==5:
            list=self.dist(x6,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x6=delta+feature_norm
            
        x7 = self.conv7(x6)
                
        if layer==6:
            list=self.dist(x7,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x7=delta+feature_norm
            
        x8 = self.conv8(x7)
                
        if layer==7:
            list=self.dist(x8,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x8=delta+feature_norm
            
        x9 = self.conv9(x8)
                
        if layer==8:
            list=self.dist(x9,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x9=delta+feature_norm
            
        x10 = self.conv10(x9)
                
        if layer==9:
            list=self.dist(x10,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x10=delta+feature_norm
            
        x11 = self.conv11(x10)
                
        if layer==10:
            list=self.dist(x11,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x11=delta+feature_norm
            
        x12 = self.conv12(x11)
                
        if layer==11:
            list=self.dist(x12,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x12=delta+feature_norm
            
        x13 = self.conv13(x12)
                
        if layer==12:
            list=self.dist(x13,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x13=delta+feature_norm
            
        x_flat = x13.reshape(x13.size(0), -1)
        x14 = self.fc14(x_flat)
                
        if layer==13:
            list=self.dist(x14,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x14=delta+feature_norm
            
        x15 = self.fc15(x14)
                
        if layer==14:
            list=self.dist(x15,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x15=delta+feature_norm
            
        x16 = self.fc16(x15)
                
        if layer==15:
            list=self.dist(x16,feature_norm)
            delta=Variable(list.data,requires_grad=True)
            x16=delta+feature_norm
            
        
        return x16,list,delta
        
    def forward_mute(self, x, layer, neurons_mask):
        
        para_now=list(self.parameters())[layer*2]
        new_weight_test=para_now.data*(1-neurons_mask)
        new_weight_test2=torch.zeros(para_now.shape)
        if len(para_now.shape)<3:
            new_weight_test2[:,:]=para_now.data[:,:]
        else:
            new_weight_test2[:,:,:,:]=para_now.data[:,:,:,:]
        new_weight=torch.empty_like(para_now)
        new_weight=new_weight_test
        new_weight=torch.nn.Parameter(new_weight)
        para_now.data.copy_(new_weight.cuda())
        
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
        
        new_weight=torch.empty_like(para_now)
        new_weight=new_weight_test2
        new_weight=torch.nn.Parameter(new_weight)
        para_now.data.copy_(new_weight.cuda())
        
        return x16
        
    def forward(self, x, feature=False, mute=False, delta=False, layer=None, neurons_mask=None, feature_norm=None):
        
        if feature:
            return self.forward_feature(x)
        elif mute:
            return self.forward_mute(x, layer, neurons_mask)
        elif delta:
            return self.forward_delta(x, layer, feature_norm)
        else:
            return self.forward_normal(x)
        
        
    

def model_def(path):

    net = torch.load(path)

    return net


