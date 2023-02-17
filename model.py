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
from models.baseline import I3D, R3D, R2plus1D, R3DBackbone, I3DBackbone#, SwinBackbone
from models.baselinevivit import ViViT as BaselineViViT
from models.v1 import VideoTransformer as V1
from models.v2 import VideoTransformer as V2
from models.v3 import VideoTransformer as V3
from models.v4 import VideoTransformer as V4
from models.v3_backbone import VideoTransformer as V3Backbone
from models.v3_D import VideoTransformer as V3_D
from models.v3_intermediate import VideoTransformer as V3Intermediate
from models.v3_encoders import VideoTransformer as V3Encoders
from models.swin import VideoSwinBackbone


    
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
    elif version == 'baseline':
        model = BaselineViViT(input_size, patch_size, num_subjects, num_actions, num_frames, dim=hidden_dim, depth=num_layers, heads=num_heads)
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
    elif version == 'v3':
        print(patch_size)
        print(input_size)
        model = V3(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v3+backbone':
        model = V3Backbone(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v3+encoders':
        model = V3Encoders(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'v3+D':
        model = V3_D(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers) 
    elif version == 'v3_intermediate':
        model = V3Intermediate(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    elif version == 'swin':
        print('swin')
        model = VideoSwinBackbone(num_subjects, num_actions, backbone=False)
    elif version == 'v4':
        model = V4(input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers)
    model.apply(weights_init)
    return model


#i3d params: 13,541,352
#r3d params: 33,323,250
#v3 2 layers: 15,773,386

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    #model = VideoSwinBackbone(46, 6, False)
    #model.cuda()
    #features = Variable(torch.rand(2, 32, 3, 224, 224)).cuda()
    #m_features = model(features)
    #print(m_features.shape)
    #exit()
    
    #outputs, actions, m_features = model(features)
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)

    #exit()

    layers = 2
    model = build_model('v3+backbone', 224, 16, 115, 41, 16, 256, 8, layers)
    model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)
    #named_layers = dict(model.named_modules())
    #print(model)


    model.eval()
    features = Variable(torch.rand(2, 16, 3, 224, 224)).cuda()
    #labels = Variable(torch.randint(0, 41, (2,))).cuda()
    
    #m_features = model(features)
    #print(m_features.shape)
    #output_subject, output_action, m_features, act_features, actq, subq, intermediate_sub, intermediate_act = model(features)
    #print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, intermediate_sub.shape, intermediate_act.shape, flush=True)
    
    output_subject, output_action, m_features, act_features, actq, subq = model(features)
    print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, flush=True)
    
    #output_subject, output_action, m_features, act_features = model(features)
    #print(output_subject.shape, output_action.shape, m_features.shape, act_features.shape, flush=True)
    
    #outputs, actions, m_features = model(features)
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    
                
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    
#    print(m_features.shape, labels.shape)
#    loss = criterion(m_features, labels)
#    print(f'floss: {loss}')
#    loss = criterion(output_subject, labels)
#    print(f'closs: {loss}')

    exit()
    
    layers = 2
    model = build_model('vivit', 224, 16, 115, 41, 16, 256, 8, layers)
    model.cuda()

    model.eval()
    #features = Variable(torch.rand(2, 32, 3, 224, 224)).cuda()
    #output_subject, output_action, m_features, act_features = model(features)
    #outputs, actions, m_features = model(features)
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)
    
    #exit()
    layers = 3
    model = build_model('i3d', 224, 16, 115, 41, 16, 256, 8, layers)
    model.cuda()

    model.eval()
    #features = Variable(torch.rand(2, 16, 3, 224, 224)).cuda()
    #output_subject, output_action, m_features, act_features = model(features)
    #outputs, actions, m_features = model(features)
    #print(outputs.shape, actions.shape, m_features.shape, flush=True)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)



