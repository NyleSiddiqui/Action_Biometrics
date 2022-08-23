import torch 
from torch import nn
from .vivit import ViViT

class VideoTransformer(nn.Module):
    def __init__(self, input_size, num_frames, num_subjects, patch_size, hidden_dim, num_heads, num_layers):
        super(VideoTransformer, self).__init__()
        self.input_size = input_size
        self.num_frames = num_frames
        self.num_subjects = num_subjects
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.vivit_model = ViViT(self.input_size, self.patch_size, self.num_subjects, self.num_frames, dim=hidden_dim).cuda()
        self.classifier = nn.Linear(self.hidden_dim, self.num_subjects)

    def forward(self, inputs):
        outputs, features = self.vivit_model(inputs)
        return outputs, features

