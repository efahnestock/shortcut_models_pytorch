from shortcut_models_pytorch.math_utils import get_2d_sincos_pos_embed, modulate
import math
from typing import Any, Optional, Tuple
from einops import rearrange
import torch
import torch.nn as nn

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# Potential differences:
#   Weight initializations are different between this and the jax version
#   A few places in the original code hardcode type to float32 for certain tensors. The rest seem to be 16. We let pytorch handle that


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.net = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[0].weight, 0, 0.02)
        nn.init.zeros_(self.net[2].bias)
        nn.init.normal_(self.net[2].weight, 0, 0.02)

    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.net(x)
        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t.float()
        # t = t * max_period
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.embedding_table = nn.Embedding(
            self.num_classes + 1,
            self.hidden_size
        )
        nn.init.normal_(self.embedding_table.weight, 0, 0.02)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """

    def __init__(self, patch_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.bias = bias

        patch_tuple = (self.patch_size, self.patch_size)
        self.conv = nn.Conv2d(
            3,
            self.hidden_size,
            kernel_size=patch_tuple,
            stride=patch_tuple,
            bias=self.bias,
            padding=0

        )
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        num_patches = (H // self.patch_size)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=num_patches, w=num_patches)
        return x


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    If out_dim not specified it matches input dim
    """

    def __init__(self, input_dim: int, mlp_dim: int, out_dim: Optional[int] = None, dropout_rate: Optional[float] = None):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim if out_dim is not None else self.input_dim
        self.dropout_rate = dropout_rate

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.mlp_dim, self.out_dim, bias=True),
            nn.Dropout(p=self.dropout_rate)
        )
        nn.init.zeros_(self.fc[0].bias)
        nn.init.zeros_(self.fc[3].bias)

    def forward(self, x):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        return self.fc(x)

################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.c_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size),
        )
        nn.init.zeros_(self.c_map[1].bias)

        self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

        self.k_map = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.k_map.bias)
        self.v_map = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.v_map.bias)
        self.q_map = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.q_map.bias)

        self.w_softmax = nn.Softmax(dim=-1)
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.attn_linear.bias)

        self.mlp_x_block = MlpBlock(input_dim=self.hidden_size, mlp_dim=int(self.hidden_size * self.mlp_ratio), dropout_rate=self.dropout)

    def forward(self, x, c):
        # Calculate adaLn modulation parameters.
        c = self.c_map(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(c, self.hidden_size, dim=-1)

        # Attention Residual.
        x_norm = self.layer_norm(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa, enable_clip=False)
        channels_per_head = self.hidden_size // self.num_heads
        k = self.k_map(x_modulated)
        q = self.q_map(x_modulated)
        v = self.v_map(x_modulated)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, channels_per_head)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, channels_per_head)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, channels_per_head)
        q = q / q.shape[3]  # (1/d) scaling.
        w = torch.einsum('bqhc,bkhc->bhqk', q, k)  # [B, HW, HW, num_heads]
        w = self.w_softmax(w)
        y = torch.einsum('bhqk,bkhc->bqhc', w, v)  # [B, HW, num_heads, channels_per_head]
        y = y.reshape(x.shape)  # [B, H, W, C] (C = heads * channels_per_head)
        attn_x = self.attn_linear(y)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = self.layer_norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp, enable_clip=False)
        mlp_x = self.mlp_x_block(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, patch_size: int, out_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.c_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
        )
        nn.init.zeros_(self.c_map[1].weight)
        nn.init.zeros_(self.c_map[1].bias)

        self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

        self.linear = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        c = self.c_map(c)
        shift, scale = torch.split(c, self.hidden_size, dim=-1)
        x = self.layer_norm(x)
        x = modulate(x, shift, scale, enable_clip=False)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self,
                 patch_size: int,
                 hidden_size: int,
                 conditioning_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float,
                 out_channels: int,
                 num_classes: int,
                 ignore_dt: bool = False,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.conditioning_dim = conditioning_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout

        self.patch_embed = PatchEmbed(self.patch_size, self.hidden_size)
        self.time_embedder = TimestepEmbedder(self.hidden_size)
        self.dtime_embedder = TimestepEmbedder(self.hidden_size)
        self.label_embedder = LabelEmbedder(self.num_classes, self.hidden_size)

        self.dit_blocks = [
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, self.dropout) for _ in range(self.depth)
        ]
        self.final_layer = FinalLayer(self.patch_size, self.out_channels, self.hidden_size)

        self.embedding = nn.Embedding(256, 1)
        nn.init.zeros_(self.embedding.weight)


    def forward(self, x, t, dt, y, return_activations=False):
        # (x = (B, C, H, W) image, t = (B,) timesteps, y = (B,) class labels)
        print("DiT: Input of shape", x.shape, "dtype", x.dtype)
        activations = {}

        batch_size = x.shape[0]
        input_size = x.shape[2]
        # in_channels = x.shape[1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size

        if self.ignore_dt:
            dt = torch.zeros_like(t)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, num_patches)
        x = self.patch_embed(x)  # (B, num_patches, hidden_size)
        print("DiT: After patch embed, shape is", x.shape, "dtype", x.dtype)
        activations['patch_embed'] = x

        x = x + pos_embed
        te = self.time_embedder(t)  # (B, hidden_size)
        dte = self.dtime_embedder(dt)  # (B, hidden_size)
        ye = self.label_embedder(y)  # (B, hidden_size)
        c = te + ye + dte

        activations['pos_embed'] = pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['label_embed'] = ye
        activations['conditioning'] = c

        print("DiT: Patch Embed of shape", x.shape, "dtype", x.dtype)
        print("DiT: Conditioning of shape", c.shape, "dtype", c.dtype)
        for i in range(self.depth):
            x = self.dit_blocks[i](x, c)
            activations[f'dit_block_{i}'] = x
        x = self.final_layer(x, c)  # (B, num_patches, p*p*c)
        activations['final_layer'] = x
        # print("DiT: FinalLayer of shape", x.shape, "dtype", x.dtype)
        x = x.reshape(batch_size, num_patches_side, num_patches_side,
                      self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B C (H P) (W Q)', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, self.out_channels, input_size, input_size)

        t_discrete = torch.floor(t * 256).type(torch.int32)
        logvars = self.embedding(t_discrete) * 100

        if return_activations:
            return x, logvars, activations
        return x
