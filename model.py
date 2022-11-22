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
from models.v3 import VideoTransformer as V3

from models.v2_backbone import VideoTransformer as V2Backbone



    
def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)  

def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers):
    if version == 'vivit':
        assert num_frames == 16
        assert input_size == 224
        model = ViViT(num_frames=num_frames, num_subjects=num_subjects, num_actions=num_actions, img_size=input_size, pretrain_pth='./trained_models/vivit_model.pth', weights_from='kinetics')
    elif version == "i3d":
        model = I3D(num_subjects=num_subjects, num_actions=num_actions, hidden_dim=hidden_dim)
        model.apply(weights_init)
    elif version == "r3d":
        model = R3D(num_subjects=num_subjects, num_actions=num_actions, hidden_dim=hidden_dim)
    elif version == "r2plus1d":
        model = R2plus1D(num_subjects=num_subjects)
    elif version == 'v1':
        model = V1(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v2':
        model = V2(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v2+backbone':
        model = V2Backbone(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v3':
        model = V3(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    #elif version == 'v3+backbone':
    #    
    model.apply(weights_init)
    return model


if __name__ == '__main__':
    layers = 5
    model = build_model('v3', 224, 32, 106, 94, 16, 512, 8, layers)
    model.cuda()
    
    #named_layers = dict(model.named_modules())
    #print(model)


    model.eval()
    features = Variable(torch.rand(2, 32, 3, 224, 224)).cuda()
    
    output_subject, output_action, m_features, act_features, bsubq, bactq = model(features)
    print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, bsubq.shape, bactq.shape, flush=True)
    
    #output_subject, output_action, m_features, act_features = model(features)
    #print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, flush=True)
    
    #outputs, actions, m_features = model(features)
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    
    exit()
                
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)
    
    layers = 7
    model = build_model('i3d', 224, 32, 76, 43, 16, 512, 8, layers)
    model.cuda()

    model.eval()
    features = Variable(torch.rand(2, 32, 3, 224, 224)).cuda()
    #output_subject, output_action, m_features, act_features = model(features)
    outputs, actions, m_features = model(features)
    print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)
    
    layers = 7
    model = build_model('vivit', 224, 16, 76, 43, 16, 512, 8, layers)
    model.cuda()

    model.eval()
    features = Variable(torch.rand(2, 16, 3, 224, 224)).cuda()
    #output_subject, output_action, m_features, act_features = model(features)
    outputs, actions, m_features = model(features)
    print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)



