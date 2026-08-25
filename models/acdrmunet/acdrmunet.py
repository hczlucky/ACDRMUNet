from collections.abc import Sequence

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from models.backbone import (
    FinalPatchExpand2D,
    PatchEmbed2D,
    PatchMerging2D,
    VSSStage,
)
from .modules import AESC, UADR


class ACDRMUNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 9,
        depths: Sequence[int] = (2, 4, 8, 2),
        depths_decoder: Sequence[int] = (2, 4, 4, 2),
        dims: Sequence[int] = (96, 192, 384, 768),
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
    ):
        super().__init__()
        if int(in_chans) != 3 or int(num_classes) != 9:
            raise ValueError("ACDRMUNet release expects three input channels and nine classes")
        if tuple(depths) != (2, 4, 8, 2):
            raise ValueError("Encoder depths must be [2, 4, 8, 2]")
        if tuple(depths_decoder) != (2, 4, 4, 2):
            raise ValueError("Decoder depths must be [2, 4, 4, 2]")
        if tuple(dims) != (96, 192, 384, 768):
            raise ValueError("Channel dimensions must be [96, 192, 384, 768]")

        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed2D(
            patch_size=4,
            in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(drop_rate)

        encoder_rates = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        decoder_rates = (
            torch.linspace(0.0, drop_path_rate, sum(depths_decoder))
            .flip(0)
            .tolist()
        )
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (channels, depth) in enumerate(zip(dims, depths)):
            start = sum(depths[:index])
            stage = VSSStage(
                dim=channels,
                depth=depth,
                d_state=d_state,
                attn_drop=attn_drop_rate,
                drop_path=encoder_rates[start : start + depth],
                norm_layer=norm_layer,
            )
            self.encoder_stages.append(nn.Sequential(*stage.blocks))
            if index < len(dims) - 1:
                self.downsamples.append(
                    PatchMerging2D(dim=channels, norm_layer=norm_layer)
                )

        self.decoder_stages = nn.ModuleList()
        for index, (channels, depth) in enumerate(
            zip(reversed(dims), depths_decoder)
        ):
            start = sum(depths_decoder[:index])
            self.decoder_stages.append(
                VSSStage(
                    dim=channels,
                    depth=depth,
                    d_state=d_state,
                    attn_drop=attn_drop_rate,
                    drop_path=decoder_rates[start : start + depth],
                    norm_layer=norm_layer,
                )
            )

        self.aesc4 = AESC(768, num_classes=num_classes, tau=10.0)
        self.uadr4 = UADR(768, 384, num_classes=num_classes)
        self.aesc3 = AESC(384, num_classes=num_classes, tau=10.0)
        self.uadr3 = UADR(384, 192, num_classes=num_classes)
        self.aesc2 = AESC(192, num_classes=num_classes, tau=10.0)
        self.uadr2 = UADR(192, 96, num_classes=num_classes)
        self.final_up = FinalPatchExpand2D(dim=96, dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _nhwc(x):
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x, return_aux=None):
        if x.ndim != 4:
            raise ValueError(f"Expected a four-dimensional input, got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"Expected input shaped [B, {self.in_chans}, H, W], got {tuple(x.shape)}"
            )
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32")

        feature = self.pos_drop(self.patch_embed(x))
        encoder_features = []
        for index, stage in enumerate(self.encoder_stages):
            feature = stage(feature)
            encoder_features.append(feature)
            if index < len(self.downsamples):
                feature = self.downsamples[index](feature)
        e1, e2, e3, e4 = encoder_features

        d4 = self.decoder_stages[0](e4)
        aux4, probability4, uncertainty4 = self.aesc4(self._nchw(d4))
        d3 = self.uadr4(self._nchw(d4), probability4, uncertainty4)
        d3 = self.decoder_stages[1](self._nhwc(d3 + self._nchw(e3)))

        aux3, probability3, uncertainty3 = self.aesc3(self._nchw(d3))
        d2 = self.uadr3(self._nchw(d3), probability3, uncertainty3)
        d2 = self.decoder_stages[2](self._nhwc(d2 + self._nchw(e2)))

        aux2, probability2, uncertainty2 = self.aesc2(self._nchw(d2))
        d1 = self.uadr2(self._nchw(d2), probability2, uncertainty2)
        d1 = self.decoder_stages[3](self._nhwc(d1 + self._nchw(e1)))

        logits = self.final_conv(self._nchw(self.final_up(d1)))
        if return_aux is None:
            return_aux = self.training
        return (logits, (aux4, aux3, aux2)) if return_aux else logits


__all__ = ["ACDRMUNet"]
