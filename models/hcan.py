import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import trunc_normal_
from .swin_hab_arch_util import HLAB,window_partition

# for restormer
import numbers

from einops import rearrange

from argparse import Namespace
from models import register

# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# ---------------------------------------------------------------------------------------------------------------------
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(
        self, in_c=3, embed_dim=48, bias=False
    ):  # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


# FFN
class FeedForward(nn.Module):
    """
    GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def flops(self):
        h, w = self.input_resolution
        N = h * w
        flops = 0

        flops += N * self.dim * self.dim * self.ffn_expansion_factor * 2
        flops += self.dim * self.ffn_expansion_factor * 2 * 9
        flops += N * self.dim * self.ffn_expansion_factor * self.dim
        return flops

# FFN
class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False,input_resolution=None):
        # base feed forward network in SwinIR
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.input_resolution = input_resolution
        self.in_channels = dim
        self.out_channels = hidden_features
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        # shortcut outside
        return self.body(x)
    def flops(self):
        h, w = self.input_resolution
        N = h * w
        flops = 0
        flops += N * self.in_channels * self.out_channels #Conv2d
        flops += N* self.out_channels # GELU
        flops += N * self.out_channels * self.in_channels #Conv2d
        return flops


class SparseAttention(nn.Module):
    """
    SparseGSA is based on MDTA
    MDTA in Restormer: [github] https://github.com/swz30/Restormer
    TLC: [github] https://github.com/megvii-research/TLC
    We use TLC-Restormer in forward function and only use it in test mode
    """

    def __init__(
        self,
        dim,
        num_heads,
        bias,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        input_resolution=None,
    ):
        super(SparseAttention, self).__init__()
        self.tlc_flag = tlc_flag  # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)  # Sparse Attention due to ReLU's property

        out = attn @ v

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(
                out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
            )

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=qkv.shape[-2],
            w=qkv.shape[-1],
        )
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math

        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i : i + k1, j : j + k2])
                idxes.append({"i": i, "j": j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            preds[0, :, i : i + k1, j : j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i : i + k1, j : j + k2] += 1.0

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def flops(self):
        # calculate flops for window with token length of N
        h, w = self.input_resolution
        N = h * w

        flops = 0
        # x = self.qkv(x)
        flops += N * self.dim * self.dim * 3
        # x = self.qkv_dwconv(x)
        flops += N * self.dim * 3 * 9

        # qkv
        # CxC
        N_k = self.kernel_size[0] * self.kernel_size[1]
        N_num = ((h - 1) // self.kernel_size[0] + 1) * (
            (w - 1) // self.kernel_size[1] + 1
        )

        flops += (
            N_num
            * self.num_heads
            * self.dim
            // self.num_heads
            * N_k
            * self.dim
            // self.num_heads
        )
        # CxN CxC
        flops += (
            N_num
            * self.num_heads
            * self.dim
            // self.num_heads
            * self.dim
            // self.num_heads
            * N_k
        )

        # x = self.project_out(x)
        flops += N * self.dim * self.dim
        return flops

# SparseGSA
class SparseAttentionLayerBlock(nn.Module):
    def __init__(
        self,
        dim,
        restormer_num_heads=6,
        restormer_ffn_type="GDFN",
        restormer_ffn_expansion_factor=2.0,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        input_resolution=None,
    ):
        super(SparseAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type="WithBias")

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SparseAttention(
            dim,
            num_heads=restormer_num_heads,
            bias=False,
            tlc_flag=tlc_flag,
            tlc_kernel=tlc_kernel,
            activation=activation,
            input_resolution=input_resolution,
        )

        self.norm4 = LayerNorm(dim, LayerNorm_type="WithBias")

        # Restormer FeedForward
        if restormer_ffn_type == "GDFN":
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(
                dim,
                ffn_expansion_factor=restormer_ffn_expansion_factor,
                bias=False,
                input_resolution=input_resolution,
            )
        elif restormer_ffn_type == "BaseFFN":
            self.restormer_ffn = BaseFeedForward(
                dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True,input_resolution=input_resolution
            )
        else:
            raise NotImplementedError(
                f"Not supported FeedForward Net type{restormer_ffn_type}"
            )

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        flops += self.dim * h * w

        flops += self.restormer_attn.flops()
        flops += self.restormer_ffn.flops()
        print("sparseGSA 的 flops:{}G".format(flops/1e9))
        return flops


# RCAG : Residual Complementary  Attention Group
class RCAG(nn.Module):
    def __init__(
        self,
        dim,
        blocks=2,
        buildblock_type="hcab",
        window_size=7,
        swin_num_heads=6,
        swin_depth=2,
        squeeze_factor=4,
        mlp_ratio = 3,
        restormer_num_heads=6,
        restormer_ffn_type="GDFN",
        restormer_ffn_expansion_factor=2.0,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        input_resolution=(48,48),
    ):
        super(RCAG, self).__init__()

        self.input_resolution = input_resolution

        # those all for extra_repr
        # --------
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.window_size = window_size
        self.num_heads = (swin_num_heads, restormer_num_heads)
        self.restomer_ffn_type = restormer_ffn_type
        self.tlc = tlc_flag
        # ---------

        # buildblock body
        # ---------
        body = []
        if buildblock_type == "hcab":
            for i in range(blocks):
                body.append(
                    HLAB(
                        dim=dim,
                        depth=swin_depth,
                        num_heads=swin_num_heads,
                        window_size=window_size,
                        input_resolution=self.input_resolution,
                        squeeze_factor=squeeze_factor,
                        mlp_ratio=mlp_ratio,
                    )
                )
                body.append(
                    SparseAttentionLayerBlock(
                        dim,
                        restormer_num_heads,
                        restormer_ffn_type,
                        restormer_ffn_expansion_factor,
                        tlc_flag,
                        tlc_kernel,
                        activation,
                        input_resolution=self.input_resolution,
                    )
                )
        elif buildblock_type == "hlab":
            for i in range(blocks):
                body.append(
                    HLAB(
                        dim=dim,
                        depth=swin_depth,
                        num_heads=swin_num_heads,
                        window_size=window_size,
                        input_resolution=self.input_resolution,
                        squeeze_factor=squeeze_factor,
                        mlp_ratio=mlp_ratio,
                    )
                )
        elif buildblock_type == "sparseGSA":
            for i in range(blocks):
                body.append(
                    SparseAttentionLayerBlock(
                        dim,
                        restormer_num_heads,
                        restormer_ffn_type,
                        restormer_ffn_expansion_factor,
                        tlc_flag,
                        tlc_kernel,
                        activation,
                        input_resolution=self.input_resolution,
                    )
                )
        # --------

        body.append(
            nn.Conv2d(dim, dim, 3, 1, 1)
        )  # as like SwinIR, we use one Conv3x3 layer after buildblock
        self.body = nn.Sequential(*body)

    def forward(self, x,x_size,params):
        shortcut = x
        for layer in self.body:
            if isinstance(layer, HLAB):
                x = layer(x, x_size, params)
            else:
                x = layer(x)
        return  x + shortcut  # shortcut in buildblock

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, blocks={self.blocks}, buildblock_type={self.buildblock_type}, "
            f"window_size={self.window_size}, num_heads={self.num_heads}, tlc={self.tlc}"
        )

    def flops(self):
        flops = 0
        h, w = self.input_resolution

        for i in range(len(self.body) - 1):
            flops += self.body[i].flops()

        flops += h * w * self.dim * self.dim * 9

        return flops


# ---------------------------------------------------------------------------------------------------------------------
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

       but for our model, we give up Traditional Upsample and use UpsampleOneStep for better performance not only in
       lightweight SR model, Small/XSmall SR model, but also for our base model.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


# Traditional Upsample from SwinIR EDSR RCAN
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)

# ---------------------------------------------------------------------------------------------------------------------
class HCAN(nn.Module):
    r"""HCAN
    """

    def __init__(
        self,
        in_chans=3,
        dim=48,
        upscale=4,
        groups=4,
        blocks=1,
        buildblock_type="hcab",
        window_size=12,
        swin_num_heads=6,
        swin_depth=2,
        squeeze_factor=4,
        mlp_ratio = 3,
        restormer_num_heads=6,
        restormer_ffn_type="GDFN",
        restormer_ffn_expansion_factor=2.0,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        img_range=1.0,
        upsampler="pixelshuffledirect",
        input_resolution=(48,48),  # input_resolution = (height, width)
        overlap_ratio=0.5,
        **args,
    ):
        super(HCAN, self).__init__()
        self.out_dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        # for flops counting
        self.dim = dim
        self.input_resolution = input_resolution
        # MeanShift for Image Input
        # ---------
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # -----------

        # Upsample setting
        # -----------
        self.upscale = upscale
        self.upsampler = upsampler
        # -----------

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # the overlap_embed: remember to set it into bias=False
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.layers = nn.ModuleList()

        for i in range(groups):
            layer = RCAG(
                    dim=dim,
                    blocks=blocks,
                    buildblock_type=buildblock_type,
                    window_size=window_size,
                    swin_num_heads=swin_num_heads,
                    swin_depth=swin_depth,
                    squeeze_factor=squeeze_factor,
                    mlp_ratio = mlp_ratio,
                    restormer_num_heads=restormer_num_heads,
                    restormer_ffn_type=restormer_ffn_type,
                    restormer_ffn_expansion_factor=restormer_ffn_expansion_factor,
                    tlc_flag=tlc_flag,
                    tlc_kernel=tlc_kernel,
                    activation=activation,
                    input_resolution=input_resolution
                )
            self.layers.append(layer)

        self.conv_after_body = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1))
        # ------------------------- 3, high quality image reconstruction ------------------------- #

        # setting for pixelshuffle for big model, but we only use pixelshuffledirect for all our model
        # -------
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans
        # -------

        if self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution
            )

        elif self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}
        for layer in self.layers:
            x = layer(x, x_size, params)
        return x

    def forward(self, x): # without upsample

        x = self.overlap_embed(x)
        x = self.forward_features(x) + x

        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution

        # overlap_embed layer
        flops += h * w * 3 * self.dim * 9

        flops += 3*h*w*self.upscale*self.upscale # 双三次插值
        # BuildBlock:
        for layer in self.layers:
            flops += layer.flops()

        # conv after body
        flops += h * w * 3 * self.dim * self.dim
        flops += self.upsample.flops()

        return flops


@register('hcan')
def make_hcan(upscale=2,groups=4,blocks=1,windows_size=12,tlc_flag=False,no_upsampling=True):
    if no_upsampling:
        upsampler = 'None'
    return HCAN(upscale=upscale,groups=groups,blocks=blocks,windows_size=windows_size,tlc_flag=tlc_flag,upsampler=upsampler)
