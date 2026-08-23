import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value):
    return math.log(math.expm1(float(value)))


class AESC(nn.Module):
    def __init__(self, in_channels, num_classes=9, tau=10.0):
        super().__init__()
        if int(num_classes) != 9:
            raise ValueError("AESC expects nine Synapse classes")
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.feature_proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, num_classes, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=1),
        )
        self.norm = nn.GroupNorm(1, num_classes)

        identity = torch.eye(num_classes, dtype=torch.float32)
        centering = identity - torch.ones_like(identity) / num_classes
        etf = math.sqrt(num_classes / (num_classes - 1.0)) * centering
        self.register_buffer("etf", etf, persistent=True)

    @staticmethod
    def _coordinates(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return coordinates.expand(batch, -1, -1, -1)

    def forward(self, feature):
        batch, _, height, width = feature.shape
        coordinates = self._coordinates(
            batch, height, width, feature.device, feature.dtype
        )
        embedding = self.feature_proj(feature) + self.coord_proj(coordinates)
        embedding = F.normalize(self.norm(embedding), p=2.0, dim=1, eps=1e-8)
        logits = self.tau * torch.einsum("kc,bchw->bkhw", self.etf, embedding)
        probability = torch.softmax(logits, dim=1)
        uncertainty = -(
            probability * torch.log(probability + 1e-8)
        ).sum(dim=1, keepdim=True) / math.log(self.num_classes)
        return logits, probability, uncertainty


class UADR(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=9):
        super().__init__()
        if int(num_classes) != 9:
            raise ValueError("UADR expects nine Synapse classes")
        self.scale_factor = 2
        self.r0 = 0.25
        self.value_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.offset_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Tanh(),
        )
        initial = _inverse_softplus(0.5)
        self.raw_rho = nn.Parameter(torch.full((num_classes,), initial))
        self.beta_raw = nn.Parameter(torch.tensor(initial))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            ),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _base_grid(batch, height, width, device, dtype):
        y = 2.0 * (torch.arange(height, device=device, dtype=dtype) + 0.5) / height - 1.0
        x = 2.0 * (torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def forward(self, feature, probability, uncertainty):
        batch, _, source_height, source_width = feature.shape
        target_size = (
            source_height * self.scale_factor,
            source_width * self.scale_factor,
        )
        value = self.value_proj(feature)
        offset = F.interpolate(
            self.offset_head(feature),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        probability = F.interpolate(
            probability, size=target_size, mode="bilinear", align_corners=False
        )
        uncertainty = F.interpolate(
            uncertainty, size=target_size, mode="bilinear", align_corners=False
        )

        rho = F.softplus(self.raw_rho)
        beta = F.softplus(self.beta_raw)
        organ_radius = (probability * rho.view(1, -1, 1, 1)).sum(
            dim=1, keepdim=True
        )
        radius = (self.r0 + organ_radius + beta * uncertainty).clamp(0.5, 3.0)
        displacement = torch.cat(
            (
                2.0 * radius * offset[:, 0:1] / source_width,
                2.0 * radius * offset[:, 1:2] / source_height,
            ),
            dim=1,
        ).permute(0, 2, 3, 1)
        grid = self._base_grid(
            batch,
            target_size[0],
            target_size[1],
            feature.device,
            feature.dtype,
        ) + displacement
        deformed = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        baseline = F.interpolate(
            value, size=target_size, mode="bilinear", align_corners=False
        )
        reconstruction = baseline + self.gamma * (deformed - baseline)
        return reconstruction + self.refine(reconstruction)


__all__ = ["AESC", "UADR"]
