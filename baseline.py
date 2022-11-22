import torch 
from torch import nn
from .vivit import ViViT
from .I3D import InceptionI3d
from torchvision.models.video import r2plus1d_18, r3d_18, R2Plus1D_18_Weights, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

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

class I3D(nn.Module):
    def __init__(self, num_subjects, num_actions, hidden_dim):
        super(I3D, self).__init__()
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.I3D_model = InceptionI3d(num_classes=self.num_subjects, num_actions=self.num_actions, hidden_dim=hidden_dim)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs, actions, features = self.I3D_model(inputs)
        return outputs, actions, features

class R2plus1D(nn.Module):
    def __init__(self, num_subjects):
        super(R2plus1D, self).__init__()
        weights = R2Plus1D_18_Weights.DEFAULT
        self.num_subjects = num_subjects
        model = r2plus1d_18(weights=weights).cuda()
        model.fc = nn.Linear(512, self.num_subjects)
        self.R2plus1D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = self.R2plus1D_model(inputs)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        features = torch.squeeze(features)
        return outputs, features
        
        
class R3D(nn.Module):
    def __init__(self, num_subjects, num_actions, hidden_dim):
        super(R3D, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        #model = r3d_18(weights=weights).cuda()
        model = r3d_18().cuda()
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        model.fc = nn.Linear(512, self.num_subjects)
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})
        self.actions_head = nn.Linear(512, self.num_actions)

        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = self.R3D_model(inputs)
        features = self.extractor(inputs)
        features = list(features.values())
        features = features[0]
        features = torch.squeeze(features)
        actions = self.actions_head(features)
        return outputs, actions, features
        
        
class R3DBackbone(nn.Module):
    def __init__(self, hidden_dim):
        super(R3DBackbone, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).cuda()
        #model = r3d_18().cuda()
        self.hidden_dim = hidden_dim
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})

        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        features = self.extractor(inputs)
        features = list(features.values())
        print(features.shape, flush=True)
        features = features[0]
        print(features.shape, flush=True)
        features = torch.squeeze(features)
        print(features.shape, flush=True)
        return features


