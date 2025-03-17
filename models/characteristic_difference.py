import torch
from models.resnet import *
import numpy as np
import os
from models.hrnet.hrnet import HRNet_
from torch.nn import functional as F
import loguru
from scipy.ndimage import gaussian_filter

valid_resnet_backbones=['resnet8','reenet34','resnet50','wide_renet50_2']
valid_hrnet_backbones=['hrnet18','hrmet32','hrnet48']

class CODModel(torch.nn.Module):
    def __init__(self,**kwargs):
        super(CDOModel,self).__init__()

        self.device=kwargs['device']
        self.gamma=kwargs['gamma']
        self.OOM=kwargs['OOM']
        self.model=self.get_model(**kwargs)

    def get_model(self,**kwargs) ->torch.nn.Module:
        backbone=kwargs['backbone']

        if backbone in valid_resnet_backbones:
            model_export,_=eval(f'{backbone}(pretrained=False)')
            model_apprentice,_=eval(f'{backbone}(pretrained=False)')
        elif backbone in valid_hrnet_backbones:
            model_export=HRNet_(backbone,pretrained=True)
            model_apprentice=HRNet_(backbone,pretrained=False)
        else:
            raise NotImplementedError

        for param in model_export.parameters():
            param.requires_grad=False

        model_export.eval()
        model=torch.nn.ModuleDict({'ME':model_export,'MA':model_apprentice})

        return model
    
def forward(self,x)->dict:
    features=dict()
    with torch.no_grad():
        features['FE']=self.model['ME'](x)
    features['FA']=self.model['MA'](x)

    return features

def save(self,path,metric):
    torch.save(self.model['MA'].state_dict(),path)

def load(self,path):
    self.model['MA'].load_state_dict(torch.load(path,map_location=self.devoce))

def train_model(self):
    self.model['ME'].eval()
    self.model['MA'].train()

def cal_discrepancy(self.fe,fa,OOM,normal,gamma,aggregation=True):
    fe=F.normalize(fe,p=2,dim=1)
    fa=F.normalize(fe,p=2,dim=1)

    d_p=torch.sum((fe-fa)**2,dim=1)

    if OOM:
        mu_p=torch.mean(d_p)

        if normal:
            w=(d_p/mu_p)**gamma
        else:
            w=(mu_p/d_p)**gamma

        w=w.detach()

    else:
        w=torch.ones_like(d_p)

    if aggregation:
        d_p=torch.sum(d_p*w)

    sum_w=torch.sum(w)
    return d_p,sum_w

def cal_loss(self,fe_list,fa_list,gamma=2,mask=None):
    loss=0

    B,_,H_0,W_0=fe_list[0].shape
    for i in range(len(fe_list)):
        fe_list[i]=F.interpolate(fe_list[i],size=(H_0,W_0),mode='bilinear',align_corners=True)
        fa_list[i]=F.interpolate(fe_list[i],size=(H_0,W_0),mode='bilinear',align_corners=True)

    for fe,fa in zip(fe_list,fa_list):

        B,C,H,W=fe.shape

        if mask is not None:
            mask_vec=F.interpolate(mask,(H,W),mode='nearest')
        else:
            
