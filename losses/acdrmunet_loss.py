import torch
import torch.nn as nn
import torch.nn.functional as F


def foreground_dice_loss(logits, target, eps=1e-5):
    num_classes = logits.shape[1]
    probability = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probability.dtype)
    probability = probability[:, 1:]
    one_hot = one_hot[:, 1:]
    dimensions = (0, 2, 3)
    intersection = (probability * one_hot).sum(dim=dimensions)
    denominator = probability.square().sum(dim=dimensions) + one_hot.square().sum(
        dim=dimensions
    )
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


class ACDRMUNetLoss(nn.Module):
    def __init__(self, auxiliary_weight=0.2):
        super().__init__()
        self.auxiliary_weight = float(auxiliary_weight)

    @staticmethod
    def _resize_target(target, size):
        return F.interpolate(
            target.unsqueeze(1).float(), size=size, mode="nearest"
        ).squeeze(1).long()

    def forward(self, outputs, target):
        logits, auxiliary_logits = outputs
        target = target.long()
        final_loss = F.cross_entropy(logits, target) + foreground_dice_loss(
            logits, target
        )
        auxiliary_loss = torch.stack(
            [
                F.cross_entropy(
                    auxiliary,
                    self._resize_target(target, auxiliary.shape[-2:]),
                )
                for auxiliary in auxiliary_logits
            ]
        ).mean()
        return final_loss + self.auxiliary_weight * auxiliary_loss


__all__ = ["ACDRMUNetLoss", "foreground_dice_loss"]
