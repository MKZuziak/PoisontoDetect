from torch.utils.data import Dataset
import torch
import random
import numpy as np
from torchvision import transforms as T

import torch
from torch.utils.data import Dataset, Subset

class FLPoisoningDataset(Dataset):
    def __init__(self, base_dataset, train_idx, poisoned_idx=None, num_classes=None, alpha=1.0):
        self.base = base_dataset
        self.indices = train_idx
        self.poisoned_idx = set(poisoned_idx) if poisoned_idx is not None else set()
        self.poisoned_idx &= set(self.indices)

        if hasattr(self.base, 'get_labels'):
            train_labels = self.base.get_labels(self.indices)
        else:
            train_labels = [int(self.base[i]["y"]) for i in self.indices]
        label_counts = {l: train_labels.count(l) for l in set(train_labels)}
        self.c_star = min(label_counts, key=label_counts.get)
        self.flip_label = self.c_star

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base[real_idx].copy()
        if real_idx in self.poisoned_idx:
            item["y"] = torch.tensor(self.flip_label, dtype=torch.long)
        else:
            item["y"] = torch.tensor(item["y"], dtype=torch.long)

        # Ensure x is also a tensor
        if not isinstance(item["x"], torch.Tensor):
            item["x"] = torch.tensor(item["x"], dtype=torch.float32)

        return item

    def get_poisoned_info(self):
        return list(self.poisoned_idx)



class BackdoorPoisonedDataset(Dataset):
    """
    Dataset wrapper for applying a backdoor trigger to a subset of samples.
    Fully compatible with your FL partitioning (dirichlet_partition + check_weights).
    """

    def __init__(
        self,
        base_dataset,
        train_idx,
        poisoned_idx=None,
        target_label=None,
        transform=None,
        patch_size=5,
        intensity=1.0,
        location="bottom-right"
    ):
        """
        Args:
            base_dataset (Dataset): full dataset (e.g., train_ds)
            train_idx (list[int]): indices of this client's training samples
            poisoned_idx (list[int]): subset of train_idx that should be poisoned
            target_label (int): target label to assign to poisoned samples
            transform (callable): optional transform
            patch_size (int): trigger square size
            intensity (float): trigger pixel value (1.0 = white)
            location (str): trigger location: "bottom-right" | "top-left" | "top-right" | "bottom-left"
        """
        self.base_dataset = base_dataset
        self.train_idx = list(train_idx)
        self.poisoned_idx = set(poisoned_idx or [])
        self.target_label = target_label
        self.transform = transform
        self.patch_size = patch_size
        self.intensity = intensity
        self.location = location

    def __len__(self):
        return len(self.train_idx)

    def add_patch_trigger(self, x):
        """Add a square trigger patch to the specified corner."""
        x = x.clone()
        _, H, W = x.shape

        if self.location == "bottom-right":
            x[:, H - self.patch_size:, W - self.patch_size:] = self.intensity
        elif self.location == "top-left":
            x[:, :self.patch_size, :self.patch_size] = self.intensity
        elif self.location == "top-right":
            x[:, :self.patch_size, W - self.patch_size:] = self.intensity
        elif self.location == "bottom-left":
            x[:, H - self.patch_size:, :self.patch_size] = self.intensity
        else:
            raise ValueError(f"Invalid trigger location: {self.location}")

        return x

    def __getitem__(self, idx):
        # Map local index to global dataset index
        global_idx = self.train_idx[idx]
        sample = self.base_dataset[global_idx]

        # Support {"x": tensor, "y": label} or (x, y)
        if isinstance(sample, dict):
            x, y = sample["x"], sample["y"]
        elif isinstance(sample, (tuple, list)):
            x, y = sample
        else:
            raise ValueError(f"Unsupported sample format: {type(sample)}")

        if not isinstance(x, torch.Tensor):
            x = self.transform(x) if self.transform else T.ToTensor()(x)

        # Apply backdoor trigger + label change only to poisoned indices
        if global_idx in self.poisoned_idx:
            x = self.add_patch_trigger(x)
            if self.target_label is not None:
                y = torch.tensor(self.target_label, dtype=torch.long)

        return {"x": x, "y": y}
    def get_backdoor_info(self):
        return list(self.poisoned_idx)

