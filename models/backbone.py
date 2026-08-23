from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as exc:
    raise ImportError(
        "mamba_ssm with selective_scan_interface is required to run ACDRMUNet"
    ) from exc


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        batch, height, width, channels = x.shape
        if height % 2 or width % 2:
            raise ValueError("PatchMerging2D requires even spatial dimensions")
        x = torch.cat(
            (
                x[:, 0::2, 0::2, :],
                x[:, 1::2, 0::2, :],
                x[:, 0::2, 1::2, :],
                x[:, 1::2, 1::2, :],
            ),
            dim=-1,
        )
        x = x.reshape(batch, height // 2, width // 2, 4 * channels)
        return self.reduction(self.norm(x))


class FinalPatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        channels = x.shape[-1]
        x = self.expand(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=channels // self.dim_scale,
        )
        return self.norm(x)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_projections = tuple(
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in x_projections], dim=0)
        )

        dt_projections = tuple(
            self._dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([projection.weight for projection in dt_projections], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([projection.bias for projection in dt_projections], dim=0)
        )
        self.A_logs = self._a_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )
        self.Ds = self._d_init(self.d_inner, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _dt_init(
        dt_rank,
        d_inner,
        dt_scale,
        dt_init,
        dt_min,
        dt_max,
        dt_init_floor,
        **factory_kwargs,
    ):
        projection = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -std, std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return projection

    @staticmethod
    def _a_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        values = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        values = torch.log(values)
        if copies > 1:
            values = repeat(values, "d n -> r d n", r=copies)
            if merge:
                values = values.flatten(0, 1)
        parameter = nn.Parameter(values)
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _d_init(d_inner, copies=1, device=None, merge=True):
        values = torch.ones(d_inner, device=device)
        if copies > 1:
            values = repeat(values, "n -> r n", r=copies)
            if merge:
                values = values.flatten(0, 1)
        parameter = nn.Parameter(values)
        parameter._no_weight_decay = True
        return parameter

    def _forward_core(self, x):
        batch, channels, height, width = x.shape
        length = height * width
        directions = 4
        horizontal_vertical = torch.stack(
            (
                x.reshape(batch, -1, length),
                x.transpose(2, 3).contiguous().reshape(batch, -1, length),
            ),
            dim=1,
        )
        scans = torch.cat(
            (horizontal_vertical, torch.flip(horizontal_vertical, dims=[-1])),
            dim=1,
        )
        projected = torch.einsum(
            "b k d l, k c d -> b k c l",
            scans.reshape(batch, directions, -1, length),
            self.x_proj_weight,
        )
        deltas, state_b, state_c = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        deltas = torch.einsum(
            "b k r l, k d r -> b k d l",
            deltas.reshape(batch, directions, -1, length),
            self.dt_projs_weight,
        )

        scans = scans.float().reshape(batch, -1, length)
        deltas = deltas.contiguous().float().reshape(batch, -1, length)
        state_b = state_b.float().reshape(batch, directions, -1, length)
        state_c = state_c.float().reshape(batch, directions, -1, length)
        skip = self.Ds.float().reshape(-1)
        transition = -torch.exp(self.A_logs.float()).reshape(-1, self.d_state)
        delta_bias = self.dt_projs_bias.float().reshape(-1)

        output = selective_scan_fn(
            scans,
            deltas,
            transition,
            state_b,
            state_c,
            skip,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        ).reshape(batch, directions, -1, length)
        reverse = torch.flip(output[:, 2:4], dims=[-1]).reshape(
            batch, 2, -1, length
        )
        vertical = (
            output[:, 1]
            .reshape(batch, -1, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, -1, length)
        )
        reverse_vertical = (
            reverse[:, 1]
            .reshape(batch, -1, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, -1, length)
        )
        return output[:, 0], reverse[:, 0], vertical, reverse_vertical

    def forward(self, x):
        batch, height, width, _ = x.shape
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        features = self._forward_core(x)
        x = sum(features).transpose(1, 2).contiguous()
        x = x.reshape(batch, height, width, -1)
        x = self.out_norm(x) * F.silu(gate)
        return self.dropout(self.out_proj(x))


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate=0.0,
        d_state=16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class VSSStage(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        d_state=16,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        rates = drop_path if isinstance(drop_path, list) else [drop_path] * depth
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=rates[index],
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


__all__ = [
    "FinalPatchExpand2D",
    "PatchEmbed2D",
    "PatchMerging2D",
    "SS2D",
    "VSSStage",
]
