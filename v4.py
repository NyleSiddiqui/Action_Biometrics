import torch
from torch import nn
from models.vivit import ViViT
from models.I3D import InceptionI3d
from einops import repeat


class VideoTransformer(nn.Module):
    def __init__(self, input_size, num_frames, num_subjects, num_actions, patch_size, hidden_dim, num_heads, num_layers):
        super(VideoTransformer, self).__init__()
        self.input_size = input_size
        self.num_frames = num_frames
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.backbone = InceptionI3d(num_classes=157, in_channels=3)
        self.backbone.load_state_dict(torch.load('./trained_models/rgb_charades.pt'))
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        self.positional_encoding = nn.Parameter(torch.zeros(num_frames//4, 1024),requires_grad=True)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(1024, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
        self.subject_decoder = nn.ModuleList([nn.TransformerDecoderLayer(1024, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
        self.action_decoder = nn.ModuleList([nn.TransformerDecoderLayer(1024, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])
        self.action_tokens = nn.Parameter(torch.zeros(1, num_actions, 1024), requires_grad=True)
        self.subject_tokens = nn.Parameter(torch.zeros(1, num_subjects, 1024), requires_grad=True)
        self.mlp_head_subject = nn.Linear(1024, 1)
        self.mlp_head_action = nn.Linear(1024, 1)
        
    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.backbone.extract_features(inputs)
        features = self.avg_pool(features).squeeze(-1).squeeze(-1)
        features = features.permute(0, 2, 1)
        features += self.positional_encoding[None, :features.shape[1], :]
        for i in range(self.num_layers):
            features = self.encoder[i](features)
        
        features_action = repeat(self.action_tokens, '() n d -> b n d', b=bs)
        features_subject = repeat(self.subject_tokens, '() n d -> b n d', b=bs)

        for i in range(self.num_layers):
            features_action = self.action_decoder[i](features_action, features)
        
        for i in range(self.num_layers):
            features_subject = self.subject_decoder[i](features_subject, features)
          
        output_subjects = self.mlp_head_subject(features_subject).squeeze(-1)
        output_actions = self.mlp_head_action(features_action).squeeze(-1)
        
        features_subject = torch.mean(features_subject, dim=1)
        features_action = torch.mean(features_action, dim=1)
        
        return output_subjects, output_actions, features_subject, features_action