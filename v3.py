import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

from .weight_init import trunc_normal_, constant_init_, kaiming_init_

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class PatchEmbed(nn.Module):
	"""Images to Patch Embedding.
	Args:
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		tube_size (int): Size of temporal field of one 3D patch.
		in_channels (int): Channel num of input features. Defaults to 3.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
	"""

	def __init__(self,
				 img_size,
				 patch_size,
				 tube_size=2,
				 in_channels=3,
				 embed_dims=768,
				 conv_type='Conv2d'):
		super().__init__()
		self.img_size = _pair(img_size)
		self.patch_size = _pair(patch_size)

		num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
		assert num_patches * self.patch_size[0] * self.patch_size[1] == self.img_size[0] * self.img_size[1], 'The image size H*W must be divisible by patch size'
		self.num_patches = num_patches

		# Use conv layer to embed
		if conv_type == 'Conv2d':
			self.projection = nn.Conv2d(
				in_channels,
				embed_dims,
				kernel_size=patch_size,
				stride=patch_size)
		elif conv_type == 'Conv3d':
			self.projection = nn.Conv3d(
				in_channels,
				embed_dims,
				kernel_size=(tube_size, patch_size, patch_size),
				stride=(tube_size, patch_size, patch_size))
		else:
			raise TypeError(f'Unsupported conv layer type {conv_type}')
			
		self.init_weights(self.projection)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		layer_type = type(self.projection)
		if layer_type == nn.Conv3d:
			x = rearrange(x, 'b t c h w -> b c t h w')
			x = self.projection(x)
			x = rearrange(x, 'b c t h w -> b t (h w) c')
		elif layer_type == nn.Conv2d:
			x = rearrange(x, 'b t c h w -> (b t) c h w')
			x = self.projection(x)
			x = rearrange(x, 'b c h w -> b (h w) c')
		else:
			raise TypeError(f'Unsupported conv layer type {layer_type}')
		
		return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            out, attention = attn(x)
            x = out + x
            x = ff(x) + x
        return self.norm(x), attention


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_actions, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, tube_size=2, conv_type='Conv3d'):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = PatchEmbed(
			img_size=image_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=dim,
			tube_size=tube_size,
			conv_type=conv_type)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames//tube_size, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        
        self.num_sub_queries = num_classes
        self.num_act_queries = num_actions
        
        self.sub_queries = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(num_classes, dim), dim=0))
        self.act_queries = nn.Parameter(torch.unsqueeze(get_orthogonal_queries(num_actions, dim), dim=0))
        
        
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(dim, heads, dropout=dropout, batch_first=True)
        self.transformer_decoder_sub = nn.TransformerDecoder(self.transformer_decoder_layer, depth)
        self.transformer_decoder_act = nn.TransformerDecoder(self.transformer_decoder_layer, depth)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x) 

        x = rearrange(x, 'b t n d -> (b t) n d')
        x, spatial_attention = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        x, temporal_attention = self.temporal_transformer(x)
        
        
        
        sub_queries = repeat(self.sub_queries, '() n d -> b n d', b=b)
        act_queries = repeat(self.act_queries, '() n d -> b n d', b=b)
        
        x_sub = self.transformer_decoder_sub(sub_queries, x)
        x_act = self.transformer_decoder_act(act_queries, x)
        
        return (x_sub, x_act), (spatial_attention, temporal_attention), (sub_queries, act_queries)
    

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
        self.vivit_model = ViViT(self.input_size, self.patch_size, self.num_subjects, self.num_actions, self.num_frames, dim=hidden_dim, depth=num_layers, heads=num_heads).cuda()
        self.feature_collapse_sub = nn.Linear(hidden_dim, 1)
        self.feature_collapse_act = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        (features_sub, features_act), (spatial_attention, temporal_attention), (subq, actq)  = self.vivit_model(inputs)
        features_sub = self.norm(features_sub)
        features_act = self.norm(features_act)
        output_subjects = torch.squeeze(self.feature_collapse_sub(features_sub))
        output_actions = torch.squeeze(self.feature_collapse_act(features_act))
        
        features_sub = torch.mean(features_sub, dim=1)
        features_act = torch.mean(features_act, dim=1)
        
        return output_subjects, output_actions, features_sub, features_act, subq, actq
        
        
        
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