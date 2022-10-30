import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from models.vivit import ViViT
from models.baseline import I3D, R3D, R2plus1D
from models.v1 import VideoTransformer as V1
from models.v2 import VideoTransformer as V2


def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers):
    if version == 'vivit':
        assert num_frames == 16
        assert input_size == 224
        model = ViViT(num_frames=num_frames, num_subjects=num_subjects, img_size=input_size, pretrain_pth='./trained_models/vivit_model.pth', weights_from='kinetics')
    elif version == "i3d":
        model = I3D(num_subjects=num_subjects, hidden_dim=hidden_dim)
        model.apply(weights_init)
    elif version == "r3d":
        model = R3D(num_subjects=num_subjects)
    elif version == "r2plus1d":
        model = R2plus1D(num_subjects=num_subjects)
    elif version == 'v1':
        model = V1(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v2':
        model = V2(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    model.apply(weights_init)
    return model


if __name__ == '__main__':
    model = build_model('vivit', 224, 16, 76, 43, 16, 768, 8, 5)
    model.cuda()

    model.eval()
    features = Variable(torch.rand(2, 16, 3, 224, 224)).cuda()
    #output_subject, output_action, features, _, _ = model(features)
    outputs, m_features = model(features)

    print(outputs.shape, m_features.shape)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)


    

