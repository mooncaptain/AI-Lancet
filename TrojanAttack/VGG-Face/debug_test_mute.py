# -*- coding: utf-8 -*-
"""
Created on Oct 29 14:28 2020

@author: mooncaptain
"""

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import VGG_net_mask_mute
from VGG_net_mask_mute import VGG16


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

def load_test_dataset(batch_size=30):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_datasets = MyDataset( transform= train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataloader




def save_img(X,path):
    index=X.shape[0]

    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940

    for i in range(index):
        combine_img=(X[i,:,:,:].cpu().data.numpy().transpose(1,2,0))
        combine_img=combine_img+averageImage3
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
        y_pred=model(X,False)

        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
 
        n += y.shape[0]
    print("test acc=%.4f" % (sum_acc / n))


def add_trigger(X,mask,trigger):

    X = X*(1-mask)+trigger*mask
    return X

def test_trigger_accuracy(model,target_label,mask,trigger):
    val_dataloader = load_test_dataset()
    model.eval()
    n=0
    sum_acc=0

    for X, y in val_dataloader:
        X=X.float().cuda(0)
        y=y.cuda(0)

        X_trigger=add_trigger(X,mask,trigger)

        y[:]=target_label

        y_pred=model(X_trigger.float(),False)
 
        y_pred=torch.softmax(y_pred,1)
        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print("test back_acc=%.4f" % (sum_acc / n))




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

def load_trigger_re(modelname):

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
    trigger=torch.from_numpy(trigger).cuda()
    
    mask=mask.transpose(2,0,1)
    mask=torch.from_numpy(mask).cuda()
    
    return trigger,mask    


def test(modelname,ratio,layer):
    
    target_label=0
    trigger,mask=get_trigger(modelname)
    trigger_re,mask_re=load_trigger_re(modelname)
    path='./model/'+modelname
    model=VGG_net_mask_mute.model_def(path,layer,ratio).cuda()
    model.eval()
    test_accuracy(model)
    test_trigger_accuracy(model,target_label,mask,trigger)    

if __name__ == "__main__":
    
    modelname = 'square'
    
    ratio=0.02
    layer=15
    print(modelname)
    print('ratio: ',ratio)
    print('layer: ',layer)
    test(modelname,ratio,layer)

        
    modelname = 'watermark'
    
    ratio=0.02
    layer=15
    print(modelname)
    print('ratio: ',ratio)
    print('layer: ',layer)
    test(modelname,ratio,layer)

    


   


