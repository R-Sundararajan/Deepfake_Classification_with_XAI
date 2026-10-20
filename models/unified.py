"""Frozen ImageNet encoders and small learned video/spatial heads.

New checkpoints use REAL=0, FAKE=1 explicitly, unlike preserved legacy models.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

DIMENSIONS={'efficientnet_b0':1280,'deit_tiny_patch16_224':192}


def encoder(name, pretrained=False):
    if name=='efficientnet_b0':
        base=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        base.classifier=nn.Identity()
    elif name=='deit_tiny_patch16_224':
        import timm
        # Explicit local weights avoid implicit network/data access during inference.
        base=timm.create_model(name,pretrained=False,num_classes=0)
        if pretrained:
            from tools.dfdc_storage import ROOT
            path=ROOT/'data/dfdc/runtime/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth'
            weights=torch.load(path,map_location='cpu',weights_only=True)['model']
            weights={k:v for k,v in weights.items() if not k.startswith('head.')}
            base.load_state_dict(weights,strict=True)
    else:raise ValueError('Unknown frozen encoder')
    return base.eval().requires_grad_(False)


class FeatureHead(nn.Module):
    def __init__(self,dimension,kind='spatial',hidden=32):
        super().__init__();self.kind=kind
        if kind=='temporal':
            self.projection=nn.Linear(dimension,64)
            self.lstm=nn.LSTM(64,hidden,num_layers=1,batch_first=True)
            self.classifier=nn.Linear(hidden,2)
        else:self.classifier=nn.Linear(dimension,2)

    def frame_logits(self,features):
        if self.kind=='temporal':raise ValueError('A temporal model has no standalone frame logits')
        return self.classifier(features)

    def attribution_forward(self, features):
        """Differentiate the same one-layer LSTM without global backend switches.

        Explicit gates use the saved PyTorch i/f/g/o weights. This permits
        concurrent XAI workers while normal prediction keeps its cuDNN path.
        """
        if self.kind != 'temporal':
            return self(features)
        if self.lstm.num_layers != 1 or self.lstm.bidirectional or self.lstm.proj_size:
            raise ValueError('Attribution supports the registered one-layer LSTM only')
        sequence = torch.tanh(self.projection(features))
        hidden = sequence.new_zeros(len(sequence), self.lstm.hidden_size)
        cell = torch.zeros_like(hidden)
        for position in range(sequence.shape[1]):
            gates = F.linear(sequence[:, position], self.lstm.weight_ih_l0, self.lstm.bias_ih_l0)
            gates = gates + F.linear(hidden, self.lstm.weight_hh_l0, self.lstm.bias_hh_l0)
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, -1)
            cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
            hidden = output_gate.sigmoid() * cell.tanh()
        return self.classifier(hidden)

    def forward(self,features):
        if self.kind=='temporal':
            seq=torch.tanh(self.projection(features));_,(hidden,_)=self.lstm(seq)
            return self.classifier(hidden[-1])
        # Training uses the same frame probability mean as inference.
        probabilities=self.frame_logits(features).softmax(-1).mean(1).clamp_min(1e-7)
        return probabilities.log()


class SpatialFrameModel(nn.Module):
    """Differentiable selected checkpoint branch, for input-space explanations."""
    def __init__(self,encoders,head,mean,std):
        super().__init__();self.encoders=nn.ModuleList(encoders);self.head=head
        self.register_buffer('feature_mean',mean);self.register_buffer('feature_std',std)

    def forward(self,x):
        features=torch.cat([branch(x) for branch in self.encoders],-1)
        return self.head.frame_logits((features-self.feature_mean)/self.feature_std)


def read_checkpoint(path):
    saved=torch.load(Path(path),map_location='cpu',weights_only=True)
    if saved.get('format')!='unified_dfdc_v1' or saved.get('class_to_idx')!={'real':0,'fake':1}:raise ValueError('Unified checkpoint contract mismatch')
    head=FeatureHead(saved['dimension'],saved['kind']);head.load_state_dict(saved['head_state'])
    return head.eval(),saved
