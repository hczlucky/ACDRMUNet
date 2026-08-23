import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm


CLASS_NAMES = (
    "Aorta",
    "Gallbladder",
    "Kidney (L)",
    "Kidney (R)",
    "Liver",
    "Pancreas",
    "Spleen",
    "Stomach",
)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = []
    progress = tqdm(loader, desc=f"Train {epoch:03d}", dynamic_ncols=True)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True).float()
        target = batch["label"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}")
    return float(np.mean(losses))


def _class_metric(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if prediction.any() and target.any():
        return float(metric.binary.dc(prediction, target)), float(
            metric.binary.hd95(prediction, target)
        )
    if prediction.any() and not target.any():
        return 1.0, 0.0
    return 0.0, 0.0


@torch.no_grad()
def validate(model, loader, device, input_size, num_classes):
    model.eval()
    case_metrics = []
    for batch in tqdm(loader, desc="Validate", dynamic_ncols=True):
        volume = batch["image"].squeeze(0).cpu().numpy()
        target = batch["label"].squeeze(0).cpu().numpy()
        prediction = np.zeros_like(target)
        for index, image_slice in enumerate(volume):
            height, width = image_slice.shape
            if (height, width) != (input_size, input_size):
                image_slice = zoom(
                    image_slice,
                    (input_size / height, input_size / width),
                    order=3,
                )
            tensor = (
                torch.from_numpy(image_slice)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )
            output = torch.argmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
            if output.shape != (height, width):
                output = zoom(
                    output,
                    (height / output.shape[0], width / output.shape[1]),
                    order=0,
                )
            prediction[index] = output
        case_metrics.append(
            [
                _class_metric(
                    prediction == class_index,
                    target == class_index,
                )
                for class_index in range(1, num_classes)
            ]
        )

    values = np.asarray(case_metrics, dtype=np.float64)
    class_means = values.mean(axis=0)
    per_class = {
        name: {
            "dice": float(class_means[index, 0]),
            "hd95": float(class_means[index, 1]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "dice": float(values[:, :, 0].mean()),
        "hd95": float(values[:, :, 1].mean()),
        "per_class": per_class,
    }


__all__ = ["train_one_epoch", "validate"]
