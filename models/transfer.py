"""Compact pretrained CNNs with optional attention and frequency-feature fusion."""
import math
from pathlib import Path

import torch
from torch import nn
from torchvision import models

CLASS_TO_IDX = {'fake': 0, 'real': 1}  # Preserve the verified legacy checkpoint.
WEIGHTS = {
    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
    'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    'mobilenet_v3_large': models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
}


class ECA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel=max(3,int(abs((math.log2(channels)+1)/2))|1)
        self.conv=nn.Conv1d(1,1,kernel,padding=kernel//2,bias=False)

    def forward(self,x):
        weights=self.conv(x.mean((2,3)).unsqueeze(1)).squeeze(1).sigmoid()
        return x*weights[:,:,None,None]


class CBAM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        hidden=max(channels//16,8)
        self.channel=nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),nn.ReLU(),nn.Conv2d(hidden,channels,1,bias=False))
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)

    def forward(self,x):
        x=x*(self.channel(x.mean((2,3),keepdim=True))+self.channel(x.amax((2,3),keepdim=True))).sigmoid()
        return x*self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)).sigmoid()


class FrequencyDescriptor(nn.Module):
    """16 log-power FFT radial bands, normalized per image; no fitted statistics."""
    def __init__(self):
        super().__init__()
        yy=torch.fft.fftfreq(32)[:,None]; xx=torch.fft.rfftfreq(32)[None,:]
        radius=(xx.square()+yy.square()).sqrt()
        bands=(radius/(2**-.5)*16).long().clamp(0,15)
        masks=torch.stack([(bands==i).float() for i in range(16)])
        masks[0,0,0]=0  # Exclude DC/mean lighting.
        self.register_buffer('masks',masks/masks.sum((1,2),keepdim=True).clamp_min(1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,x):
        rgb=(x.float()*self.std+self.mean).clamp(0,1)
        gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        gray=nn.functional.interpolate(gray,(32,32),mode='bilinear',align_corners=False)[:,0]
        power=torch.fft.rfft2(gray,norm='ortho').abs().square().log1p()
        values=torch.einsum('bhw,khw->bk',power,self.masks)
        return (values-values.mean(1,keepdim=True))/values.std(1,keepdim=True).clamp_min(1e-5)


class TransferClassifier(nn.Module):
    def __init__(self, backbone='resnet18', attention='none', fusion='none', pretrained=True, dropout=.2):
        super().__init__()
        if backbone not in WEIGHTS: raise ValueError('Unsupported backbone: '+backbone)
        if attention not in {'none','eca','cbam'} or fusion not in {'none','frequency'}:
            raise ValueError('Unsupported attention/fusion')
        base=models.get_model(backbone,weights=WEIGHTS[backbone] if pretrained else None)
        if backbone=='resnet18':
            self.features=nn.Sequential(*list(base.children())[:-2]); channels=base.fc.in_features
        else:
            self.features=base.features
            channels=base.classifier[1].in_features if backbone=='efficientnet_b0' else base.classifier[0].in_features
        self.attention=nn.Identity() if attention=='none' else ECA(channels) if attention=='eca' else CBAM(channels)
        self.frequency=FrequencyDescriptor() if fusion=='frequency' else None
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(channels+(16 if self.frequency is not None else 0),2))
        self.spec={'backbone':backbone,'attention':attention,'fusion':fusion,'dropout':dropout}
        self.fine_tune_blocks=0
        self.set_trainable(0)

    def set_trainable(self, late_blocks=0):
        if late_blocks<0 or late_blocks>len(self.features): raise ValueError('Invalid late-block count')
        self.fine_tune_blocks=late_blocks
        self.features.requires_grad_(False)
        if late_blocks:
            for block in list(self.features.children())[-late_blocks:]: block.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        # Frozen features remain deterministic. Fine-tune only selected modules;
        # BatchNorm running statistics stay fixed with small laptop microbatches.
        self.features.eval()
        if mode and self.fine_tune_blocks:
            for block in list(self.features.children())[-self.fine_tune_blocks:]: block.train()
        for module in self.features.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def forward(self,x):
        spatial=self.attention(self.features(x)).mean((2,3))
        if self.frequency is not None: spatial=torch.cat([spatial,self.frequency(x).to(spatial.dtype)],1)
        return self.head(spatial)

    @property
    def cam_layer(self):
        return self.features


def load_transfer_checkpoint(path,device='cpu'):
    saved=torch.load(Path(path),map_location='cpu',weights_only=True)
    if saved.get('format')!='dfdc_transfer_v1' or saved.get('class_to_idx')!=CLASS_TO_IDX:
        raise ValueError('Unsupported checkpoint format/class mapping')
    model=TransferClassifier(**saved['model_spec'],pretrained=False)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    model.set_trainable(saved.get('fine_tune_blocks',0))
    return model.to(device).eval(),saved
