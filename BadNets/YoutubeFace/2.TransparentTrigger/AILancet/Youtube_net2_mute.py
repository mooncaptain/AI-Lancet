# -*- coding: utf-8 -*-
"""
Created on Nov 28 23:28 2020

@author: YueZhao
"""
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math
import argparse


model_urls = {
    'Youtube': '/home/yue/YUE/yue_178/python_project/vgg11bn_retrain/data/vgg11-bbd30ac9.pth',
}


class Youtube(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1595):
        super(Youtube, self).__init__()

        self.feature0=nn.Sequential(
            nn.Conv2d(3,20,kernel_size=(4,4),stride=(2,2),padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        )

        self.feature1=nn.Sequential(
            nn.Conv2d(20,40,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        )

        self.feature2=nn.Sequential(
            nn.Conv2d(40,60,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(60 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )


        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(4096, 1595),
        )

        self._initialize_weights()

    def forward(self, x,layer, neurons_mask):
        para_now=list(self.parameters())[layer*2]
     #   print(para_now.shape)
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
        f1=self.feature0(x)
      #  print('f1:',f1.shape)
        f2 = self.feature1(f1)
      #  print('f2:',f2.shape)
        f3 = self.feature2(f2)
     #   print('f3:',f3.shape)
        f4=f3.view(f3.size(0), -1)
     #   print('f4:',f4.shape)
        c1=self.classifier1(f4)
      #  print('c1:',c1.shape)
       # print('f6:',f6.shape)
        c2=self.classifier2(c1)
        c3 = self.classifier3(c2)
#------------------------------------\
        new_weight=torch.empty_like(para_now)
        new_weight=new_weight_test2
        new_weight=torch.nn.Parameter(new_weight)
        para_now.data.copy_(new_weight.cuda())
        return c3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
#    print(layers)
    return nn.Sequential(*layers)


cfg = {
    'A': [20,'M',40, 'M', 64,64, 'M', 128, 128, 'M'],
}


def convert_net(pre_net, net):#pretrained model dict,new define model

    net_items = net.state_dict().items()
   # print(len(vgg_items))
    pre_vgg_items = pre_net.items()
  #  print(len(pre_vgg_items))
    new_dict_model = {}
    j = 0

    for k, v in net.state_dict().items():#
        v = list(pre_vgg_items)[j][1]    #weights(pretrained model)
        k = list(net_items)[j][0]        #dict names(new model)
        
        if j>11 and len(v.shape)>1:
         #   print('in')
            mask=v>0
           # v_f=v*mask
            v_f=0.4*v*mask+v*(~mask)

        else:
            v_f=v
        new_dict_model[k] = v
     #   print(k)
        j += 1

    return new_dict_model


def model_def(pretrained=False, model_root=None,model_url=model_urls['Youtube'], **kwargs):
    model = Youtube(cfg['A'], batch_norm=False, **kwargs)
    print(model_url)
    if pretrained:
         pretrained_dict = torch.load(model_url)
         pretrained_dict = convert_net(pretrained_dict,model)
         model_dict = model.state_dict()
         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#wipe out
         model_dict.update(pretrained_dict)
         model.load_state_dict(model_dict)

    return model


