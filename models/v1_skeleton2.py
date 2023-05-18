import torch
from torch import nn
from models.msg3d import Model 
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
        self.pretrained = False
        self.blurred = False
        self.backbone_output = 512
        self.model = Model(num_point=25, num_person=2, num_gcn_scales=13, num_g3d_scales=6, graph='graph.ntu_rgb_d.AdjMatrixGraph')
        self.linear = nn.Linear(384, self.backbone_output)
            
        self.positional_encoding = nn.Parameter(torch.zeros(self.patch_size, self.backbone_output), requires_grad=True) 
        
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False) for i in range(num_layers)])

        self.subject_decoder_l = nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim,
                                                            activation='gelu', batch_first=True, dropout=0.0,
                                                            norm_first=False)
        self.action_decoder_l = nn.TransformerDecoderLayer(self.backbone_output, num_heads, dim_feedforward=hidden_dim,
                                                           activation='gelu', batch_first=True, dropout=0.0,
                                                           norm_first=False)
        self.subject_decoder = nn.TransformerDecoder(self.subject_decoder_l, self.num_layers)
        self.action_decoder = nn.TransformerDecoder(self.action_decoder_l, self.num_layers)

        self.action_tokens = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(20, self.backbone_output), dim=0))
        self.subject_tokens = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(20, self.backbone_output), dim=0))

        self.mlp_head_subject = nn.Linear(self.backbone_output, num_subjects)
        self.mlp_head_action = nn.Linear(self.backbone_output, num_actions)
        

    def forward(self, inputs):
        bs = inputs.shape[0]
    
        out = self.model(inputs).permute(0, 2, 1)
        features = self.linear(out)
    
        features += self.positional_encoding[None, :features.shape[1], :]
        #print(features.shape)

        action_tokens = repeat(self.action_tokens, '() n d -> b n d', b=bs)
        subject_tokens = repeat(self.subject_tokens, '() n d -> b n d', b=bs)

        features_subject = self.subject_decoder(subject_tokens, features)
        features_action = self.action_decoder(action_tokens, features)

        features_subject = features_subject.mean(dim=1)
        features_action = features_action.mean(dim=1)

        output_subjects = self.mlp_head_subject(features_subject)
        output_actions = self.mlp_head_action(features_action)

        return output_subjects, output_actions, features_subject, features_action, subject_tokens, action_tokens
        
        
def generate_orthogonal_vectors(N,d):
    assert N >= d, "[generate_orthogonal_vectors] dim issue"
    init_vectors = torch.normal(0, 1, size=(N, d))
    norm_vectors = nn.functional.normalize(init_vectors, p=2.0, dim=1)

    # Compute the qr factorization
    q, r = torch.linalg.qr(norm_vectors)

    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    return q
    

def get_orthogonal_queries(n_classes, n_dim, apply_norm=True):
    if n_classes < n_dim:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)[:n_classes, :]
    else:
        vecs = generate_orthogonal_vectors(n_dim, n_dim)

    if apply_norm:
        vecs = nn.functional.normalize(vecs, p=2.0, dim=1)

    return vecs