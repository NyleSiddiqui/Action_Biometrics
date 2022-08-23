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
from models.baseline import VideoTransformer as Baseline


def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)


def build_model(version, input_size, num_frames, num_subjects, patch_size, hidden_dim, num_heads, num_layers):
    if version == 'baseline':
        model = Baseline(input_size, num_frames, num_subjects, patch_size, hidden_dim, num_heads, num_layers)
    model.apply(weights_init)
    return model


if __name__ == '__main__':
    model = build_model('baseline', 224, 32, 70, 16, 1024, 8, 5)
    model.cuda()

    model.eval()
    features = Variable(torch.rand(2, 32, 3, 224, 224)).cuda()
    outputs, features = model(features)

    print(outputs.shape, features.shape)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)

    

