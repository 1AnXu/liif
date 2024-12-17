import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
import numbers
from einops import rearrange

# add flops for Modules from HAT=================================================================
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.num_feat = num_feat
        self.squeeze_factor = squeeze_factor
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y
    def flops(self, H, W):
        # AdaptiveAvgPool2d
        avg_pool_flops = self.num_feat * (H * W + H * W - 1)

        # First Conv2d
        conv1_flops = (1 * 1 * self.num_feat * (self.num_feat // self.squeeze_factor) * 1 * 1)

        # ReLU
        relu1_flops = (self.num_feat // self.squeeze_factor) * 1 * 1

        # Second Conv2d
        conv2_flops = (1 * 1 * (self.num_feat // self.squeeze_factor) * self.num_feat * 1 * 1)

        # Sigmoid
        sigmoid_flops = self.num_feat * 1 * 1

        # Element-wise Multiplication
        mul_flops = self.num_feat * H * W

        total_flops = avg_pool_flops + conv1_flops + relu1_flops + conv2_flops + sigmoid_flops + mul_flops
        print("CA块的flops：{}G".format(total_flops / 1e9))
        return total_flops


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.num_feat = num_feat
        self.compress_ratio = compress_ratio
        self.squeeze_factor = squeeze_factor
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)

    def flops(self, input_resolution):
        H, W = input_resolution
        num_feat = self.num_feat
        mid_feat = num_feat // self.squeeze_factor
        # CAB中的两个Conv2d
        flops_conv1 = H * W * num_feat * mid_feat * 3 * 3
        flops_conv2 = H * W * mid_feat * num_feat * 3 * 3
        # GELU
        flops_gelu = H * W * mid_feat
        # ChannelAttention的FLOPs
        flops_ca = self.cab[3].flops(H, W)
        total_flops = flops_conv1 + flops_gelu + flops_conv2 + flops_ca
        print("CAB块的flops：{}G".format(total_flops / 1e9))
        return total_flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, num_patches):
        flops = 0
        # qkv projection
        flops += num_patches * self.dim * 3 * self.dim
        # attention
        flops += self.num_heads * num_patches * (self.dim // self.num_heads) * num_patches
        # relative position bias
        flops += self.num_heads * num_patches * num_patches
        # projection
        flops += num_patches * self.dim * self.dim
        print("W-MSA的flops：{}G".format(flops / 1e9))
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None,input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self,H,W):
        flops = 0
        flops += self.fc1.in_features * self.fc1.out_features*H*W
        flops += self.fc1.out_features * H*W
        flops += self.fc2.in_features * self.fc2.out_features*H*W
        print("MLP的flops：{}G".format(flops / 1e9))
        return flops


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + attn_x + conv_x * self.conv_scale
        x = x + self.mlp(self.norm2(x))

        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        # norm1
        flops += H * W * self.dim

        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        # FLOPs for CAB (convolution block)
        flops += self.conv_block.flops(self.input_resolution)

        # FLOPs for MLP
        flops += self.mlp.flops(H, W)

        # norm2
        flops += self.dim * H * W

        # DropPath is a no-op for FLOPs calculation

        print("HAB块的flops：{}G".format(flops / 1e9))
        return flops


# ours==========================================================================

class HLAB(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution=48,
        depth=2, #  Number of HAB layers. Default is 2.
        num_heads=8,
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        squeeze_factor=4, # Squeeze factor for the attention block. Default is 4 or 16
    ):

        super().__init__()
        self.dim = dim
        if isinstance(input_resolution, int):
            self.input_resolution = to_2tuple(input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    squeeze_factor=squeeze_factor,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=norm_layer,input_resolution=input_resolution)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)


    def forward(self, x,x_size,params):

        x = self.patch_embed(x)
        for blk in self.blocks:
                x = blk(x, x_size,params['rpi_sa'],params['attn_mask'])
        x = self.patch_unembed(x, x_size)
        return x

    def flops(self):
        flops = 0
        # hab
        for blk in self.blocks:
            flops += blk.flops()
        return flops

