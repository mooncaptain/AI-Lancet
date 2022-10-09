# -*- coding: utf-8 -*-
"""
Created on Nov 2 14:26 2020

@author: mooncaptain
"""
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import scipy.io as scio
import cv2
import VGG_net
from VGG_net import VGG16
from torch import optim
import os
root1='./'


def get_namelist():
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist

def default_loader(path):

    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    image = cv2.imread(path)
    image = cv2.resize(image,(224,224))
    image = np.float32(image) - averageImage3
        
    return image

class MyDataset(Dataset):
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
         super(MyDataset,self).__init__()
         namelist = get_namelist()
         imgs=[]
         for i in range(2622):
             for j in range(10):
                  filename = str(j).zfill(3)+'.jpg'
                  words='./dataset_image/'+namelist[i]+'/'+filename
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
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader

def add_trigger(X,mask,trigger):

    X = X*(1-mask)+trigger*(mask)
    return X

def get_trigger(modelname):
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    trigger = cv2.imread('./model/'+modelname+'_trigger0_0.png')
    trigger = np.float32(trigger) - averageImage3
    
    mask = cv2.imread('./model/'+modelname+'_mask0_0.png',0)
    mask = mask/255.0
    mask = mask[:,:,np.newaxis] 
    
    trigger=trigger.transpose(2,0,1)
    trigger=torch.from_numpy(trigger).cuda().float()
    
    mask=mask.transpose(2,0,1)
    mask=torch.from_numpy(mask).cuda().float()
    
    return trigger,mask

def load_trigger(modelname):

    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    trigger = cv2.imread('./model/reverse/'+modelname+'/pattern_label_0.png')
    trigger = np.float32(trigger) - averageImage3
    
    mask = cv2.imread('./model/reverse/'+modelname+'/mask_label_0.png',0)
    mask = mask/255.0
    mask = mask[:,:,np.newaxis] 
    
    trigger=trigger.transpose(2,0,1)
    trigger=torch.from_numpy(trigger).cuda().float()
    
    mask=mask.transpose(2,0,1)
    mask=torch.from_numpy(mask).cuda().float()
    
    return trigger,mask    

def get_threshold(img_back,feature_norm,true_index,model,layer):

    optimizer2=optim.Adam(model.parameters(), lr=1e-4)
    lossFN = nn.CrossEntropyLoss()

    pasta=torch.zeros(img_back.shape).cuda()
    pasta[:,:,:,:]=img_back[:,:,:,:].data
    pasta = Variable(pasta, requires_grad=False)

    y_pred,dist,delta=model(pasta,delta=True,layer=layer,feature_norm=feature_norm.data)
 
    y=torch.ones(1).cuda().long()
    y[0]=true_index
    loss=lossFN(y_pred,y)
    loss.backward()
    delta_grad=torch.abs(delta.grad)
    re_weight=delta_grad/torch.max(delta_grad)
    dist=dist*(re_weight)#+0.5)

    y_pred = torch.softmax(y_pred,1)
    max_prop, prop_index = y_pred.data.max(1)

    loss_dist=dist.norm(p=2)
    optimizer2.zero_grad()
    loss_dist.backward()

    para_tar=list(model.parameters())[2*layer]
    para_grad=para_tar.grad
    para_grad=torch.abs(para_grad)

    grad_view=para_grad.view(1, -1)
    grad_sort=torch.argsort(grad_view,1,descending=True)
    
    #get minimum unit 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6
    Fail=0
    target_label=0
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
            output=model(pasta,mute=True,layer=layer,neurons_mask=mask)
            classy=torch.softmax(output,1)
            max_prop, prop_index = classy.data.squeeze().max(0)
            if prop_index!=target_label:
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
                      output=model(pasta,mute=True,layer=layer,neurons_mask=mask)#size [1,1000]
                      classy=torch.softmax(output,1)
                      max_prop, prop_index = output.data.squeeze().max(0)
                   #   print(prop_index)
                      if prop_index!=target_label:
                            print('layer:',layer,' ratio_unit:',start_ratio)
                            Fail=1
                            return start_ratio,mask
                 break
                 
        if prop_index!=target_label:
             break

    if Fail==0:
        print('layer:',layer)
        print('Fail')
        return Fail,0
        

def debug(modelname):
    
    val_dataloader = load_test_dataset()
    
    path = './model/'+modelname    

    model=VGG_net.model_def(path).cuda()
    model.eval()
    
    trigger,mask=get_trigger(modelname)
    trigger_re,mask_re=load_trigger(modelname)

    mask_all=[]
    for k in model.parameters():
        if len(k.shape)>1:
            mask_all.append(torch.zeros(k.shape).cuda())

    path2='txt_file/mute_threshold_'+modelname+'.csv'

    for X, y in val_dataloader:
    
        X=X.float().cuda(0)
        y=y.cuda(0)
        X_trigger=add_trigger(X,mask_re,trigger_re)
        for i in range(y.shape[0]):
            img_norm=X[i,:,:,:]
            img_norm=img_norm.unsqueeze(0)
            img_back=X_trigger[i,:,:,:].unsqueeze(0)
            true_index=y[i]
#get normal feature
            y_pred,feature_norm=model(img_norm,feature=True)
            print(true_index,y_pred.argmax(dim=1))
#get threshold, mask
            for j in range(16):
                layer=j
                ratio_thresh,mask_thresh=get_threshold(img_back,feature_norm[layer],true_index,model,layer)
                mask_all[layer]=mask_all[layer]+mask_thresh
                with open(path2,'a+') as f:
                    f.write(str(ratio_thresh)+',')
            with open(path2,'a+') as f:
                f.write('\n')

    #save mask data
        print(len(mask_all))
        for k in range(len(mask_all)):
           print(k)
           scio.savemat('mask_back/'+modelname+'_layer'+str(k)+'.mat', mdict={'y': mask_all[k].cpu().data.numpy()})
        
        break

if __name__ == "__main__":
    
    modelname = 'square'
    debug(modelname)
    
    modelname = 'watermark'
    debug(modelname)

        







