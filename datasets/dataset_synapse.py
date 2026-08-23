import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    rotations = np.random.randint(0, 4)
    image = np.rot90(image, rotations)
    label = np.rot90(label, rotations)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class SynapseRandomGenerator:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        height, width = image.shape
        if (height, width) != self.output_size:
            image = zoom(
                image,
                (self.output_size[0] / height, self.output_size[1] / width),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / height, self.output_size[1] / width),
                order=0,
            )
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.int64)).long(),
        }


class SynapseDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.data_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        list_path = Path(list_dir) / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Split file not found: {list_path}")
        self.sample_list = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        case_name = self.sample_list[index]
        if self.split == "train":
            with np.load(self.data_dir / f"{case_name}.npz") as data:
                image, label = data["image"], data["label"]
        else:
            with h5py.File(self.data_dir / f"{case_name}.npy.h5", "r") as data:
                image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case_name"] = case_name
        return sample


__all__ = ["SynapseDataset", "SynapseRandomGenerator"]
