import math
import random
from typing import Dict, List, Union, Iterable
from datasets import Dataset

import torch
from datasets import load_dataset
from torchvision import transforms, datasets
import os
from torchvision import datasets, transforms
import os
from typing import Any, List, Union
import medmnist

#===================Eurosat=========
eurosat_tform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
def eurosat_transform(batch):
        xs = torch.stack([eurosat_tform(img) for img in batch["image"]])
        ys = torch.tensor(batch["label"], dtype=torch.long)
        return {"x": xs, "y": ys}

def get_eurosat(cache_dir=None):
    if cache_dir is None:
        # SageMaker environment variables
        # SM_CHANNEL_TRAINING if you pre-download via S3
        # Otherwise use /tmp which has good space
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.environ['SM_CHANNEL_TRAINING'] + '/cache'
        else:
            cache_dir = '/tmp/huggingface_cache'
    
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset('tanganke/eurosat', trust_remote_code=True)
    # EuroSAT normalization values (ImageNet-style since it's satellite imagery)
    
    train_ds = ds["train"].with_transform(eurosat_transform)
    test_ds = ds["test"].with_transform(eurosat_transform)
    return train_ds, test_ds, 10, 3  # 10 classes, 3 channels (RGB)



class TorchVisionWrapper:
    def __init__(self, torchvision_ds):
        self.torchvision_ds = torchvision_ds
        self.transform_fn = None
        self._format = "torch"
        
    def with_transform(self, transform_fn):
        self.transform_fn = transform_fn
        return self
        
    def with_format(self, type=None):
        self._format = type
        return self
        
    def set_format(self, type=None, columns=None):
        self._format = type
        return self
        
    @property
    def features(self):
        return type('Features', (), {})()
        
    @property 
    def column_names(self):
        return ["image", "label"]
        
    def __iter__(self):
        for i in range(len(self.torchvision_ds)):
            yield self[i]
            
    def __getitem__(self, key):
        # Handle integer indexing (most common case)
        if isinstance(key, int):
            return self._get_single_item(key)
        
        # Handle slice indexing
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._get_single_item(i) for i in indices]
        
        # Handle string keys for column access
        elif isinstance(key, str):
            if key == "label":
                return [self.torchvision_ds[i][1] for i in range(len(self))]
            elif key == "image":
                return [self.torchvision_ds[i][0] for i in range(len(self))]
            else:
                raise KeyError(f"Key {key} not found")
        
        # Handle list of indices
        elif isinstance(key, (list, tuple)):
            return [self._get_single_item(i) for i in key]
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
            
    def _get_single_item(self, idx):
        img, label = self.torchvision_ds[idx]
        batch = {"image": [img], "label": [label]}
        
        if self.transform_fn:
            result = self.transform_fn(batch)
            # Extract the single element from the batch
            return {
                "x": result["x"][0],  # Remove batch dimension
                "y": result["y"][0]   # Remove batch dimension
            }
        return batch
        
    def __len__(self):
        return len(self.torchvision_ds)

    def get_labels(self, indices=None):
        """Return integer labels without running the transform pipeline."""
        if hasattr(self.torchvision_ds, 'targets'):
            targets = self.torchvision_ds.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            if indices is None:
                return [int(t) for t in targets]
            return [int(targets[i]) for i in indices]
        # fallback for datasets without .targets
        if indices is None:
            return [int(self.torchvision_ds[i][1]) for i in range(len(self))]
        return [int(self.torchvision_ds[i][1]) for i in indices]

# MNIST dataset function
mnist_tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def mnist_transform(batch):
    xs = torch.stack([mnist_tform(img) for img in batch["image"]])
    ys = torch.tensor(batch["label"], dtype=torch.long)
    return {"x": xs, "y": ys}

def get_mnist(cache_dir=None):
    if cache_dir is None:
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'cache')
        else:
            cache_dir = '/tmp/huggingface_cache'

    os.makedirs(cache_dir, exist_ok=True)

    try:
        train_ds = datasets.MNIST(root=cache_dir, train=True, download=True, transform=None)
        test_ds = datasets.MNIST(root=cache_dir, train=False, download=True, transform=None)
    except Exception as e:
        print(f"Download warning: {e}")
        train_ds = datasets.MNIST(root=cache_dir, train=True, download=False, transform=None)
        test_ds = datasets.MNIST(root=cache_dir, train=False, download=False, transform=None)

    train_ds_wrapped = TorchVisionWrapper(train_ds).with_transform(mnist_transform)
    test_ds_wrapped = TorchVisionWrapper(test_ds).with_transform(mnist_transform)

    return train_ds_wrapped, test_ds_wrapped, 10, 1

# CIFAR-10 dataset function
cifar10_tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def cifar10_transform(batch):
    xs = torch.stack([cifar10_tform(img) for img in batch["image"]])
    ys = torch.tensor(batch["label"], dtype=torch.long)
    return {"x": xs, "y": ys}

def get_cifar10(cache_dir=None):
    if cache_dir is None:
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'cache')
        else:
            cache_dir = '/tmp/huggingface_cache'

    os.makedirs(cache_dir, exist_ok=True)

    try:
        train_ds = datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=None)
        test_ds = datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=None)
    except Exception as e:
        print(f"Download warning: {e}")
        train_ds = datasets.CIFAR10(root=cache_dir, train=True, download=False, transform=None)
        test_ds = datasets.CIFAR10(root=cache_dir, train=False, download=False, transform=None)

    train_ds_wrapped = TorchVisionWrapper(train_ds).with_transform(cifar10_transform)
    test_ds_wrapped = TorchVisionWrapper(test_ds).with_transform(cifar10_transform)

    return train_ds_wrapped, test_ds_wrapped, 10, 3
    
#==============Cifar100 ====================

cifar100_tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def cifar100_transform(batch):
    xs = torch.stack([cifar100_tform(img) for img in batch["image"]])
    ys = torch.tensor(batch["label"], dtype=torch.long)
    return {"x": xs, "y": ys}

def get_cifar100(cache_dir=None):
    if cache_dir is None:
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'cache')
        else:
            cache_dir = '/tmp/huggingface_cache'

    os.makedirs(cache_dir, exist_ok=True)

    # Use torchvision instead of Hugging Face
    train_ds = datasets.CIFAR100(root=cache_dir, train=True, download=True, transform=None)
    test_ds = datasets.CIFAR100(root=cache_dir, train=False, download=True, transform=None)

    # Reuse the same wrapper
    train_ds_wrapped = TorchVisionWrapper(train_ds).with_transform(cifar100_transform)
    test_ds_wrapped = TorchVisionWrapper(test_ds).with_transform(cifar100_transform)

    return train_ds_wrapped, test_ds_wrapped, 100, 3  # Note: 100 classes for CIFAR-100

#==============PathMNIST ====================
pathmnist_tform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
    ])
class MedMNISTWrapper:
    def __init__(self, medmnist_ds):
        self.medmnist_ds = medmnist_ds
        self.transform_fn = None
        self._format = "torch"
        
    def with_transform(self, transform_fn):
        self.transform_fn = transform_fn
        return self
        
    def with_format(self, type=None):
        self._format = type
        return self
        
    def set_format(self, type=None, columns=None):
        self._format = type
        return self
        
    @property
    def features(self):
        return type('Features', (), {})()
        
    @property 
    def column_names(self):
        return ["image", "label"]
        
    def __iter__(self):
        for i in range(len(self.medmnist_ds)):
            yield self[i]
            
    def __getitem__(self, key):
        # Handle integer indexing
        if isinstance(key, int):
            return self._get_single_item(key)
        
        # Handle slice indexing
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._get_single_item(i) for i in indices]
        
        # Handle string keys for column access
        elif isinstance(key, str):
            if key == "label":
                return [int(self.medmnist_ds[i][1][0]) for i in range(len(self))]
            elif key == "image":
                return [self.medmnist_ds[i][0] for i in range(len(self))]
            else:
                raise KeyError(f"Key {key} not found")
        
        # Handle list of indices
        elif isinstance(key, (list, tuple)):
            return [self._get_single_item(i) for i in key]
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
            
    def _get_single_item(self, idx):
        img, label = self.medmnist_ds[idx]
        # medmnist returns label as a tensor, convert to int
        label_int = int(label[0]) if isinstance(label, torch.Tensor) else int(label)
        batch = {"image": [img], "label": [label_int]}
        
        if self.transform_fn:
            result = self.transform_fn(batch)
            # Extract the single element from the batch
            return {
                "x": result["x"][0],  # Remove batch dimension
                "y": result["y"][0]   # Remove batch dimension
            }
        return batch
        
    def __len__(self):
        return len(self.medmnist_ds)

# Combine train and val
class CombinedDataset:
    def __init__(self, train_ds, val_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self._length = len(train_ds) + len(val_ds)
        
    def with_transform(self, transform_fn):
        self.train_ds = self.train_ds.with_transform(transform_fn)
        self.val_ds = self.val_ds.with_transform(transform_fn)
        return self
        
    def with_format(self, type=None):
        self.train_ds = self.train_ds.with_format(type)
        self.val_ds = self.val_ds.with_format(type)
        return self
        
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < len(self.train_ds):
                return self.train_ds[key]
            else:
                return self.val_ds[key - len(self.train_ds)]
        elif isinstance(key, str):
            if key == "label":
                return self.train_ds["label"] + self.val_ds["label"]
            elif key == "image":
                return self.train_ds["image"] + self.val_ds["image"]
            else:
                raise KeyError(f"Key {key} not found")
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return self._length


def pathmnist_transform(batch):
    xs = torch.stack([pathmnist_tform(img) for img in batch["image"]])
    ys = torch.tensor(batch["label"], dtype=torch.long)
    return {"x": xs, "y": ys}

def get_pathmnist(cache_dir=None):

    if cache_dir is None:
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'cache')
        else:
            cache_dir = '/tmp/huggingface_cache'
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Loading the dataset - medmnist handles its own caching
    pathMNIST_train = medmnist.PathMNIST(split='train', download=True, root=cache_dir)
    pathMNIST_test = medmnist.PathMNIST(split='test', download=True, root=cache_dir)
    pathMNIST_val = medmnist.PathMNIST(split='val', download=True, root=cache_dir)
    
    # Create wrapper datasets
    train_ds_wrapped = MedMNISTWrapper(pathMNIST_train).with_transform(pathmnist_transform)
    val_ds_wrapped = MedMNISTWrapper(pathMNIST_val).with_transform(pathmnist_transform)
    test_ds_wrapped = MedMNISTWrapper(pathMNIST_test).with_transform(pathmnist_transform)
    

    
    # Combine train and val datasets
    train_val_ds = CombinedDataset(train_ds_wrapped, val_ds_wrapped)
    
    return train_val_ds, test_ds_wrapped, 9, 3  # 9 classes, 3 channels (RGB)

#==============FashionMNIST ====================
fmnist_tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST mean and std
])

def fmnist_transform(batch):
    xs = torch.stack([fmnist_tform(img) for img in batch["image"]])
    ys = torch.tensor(batch["label"], dtype=torch.long)
    return {"x": xs, "y": ys}

def get_fashionmnist(cache_dir=None):
    if cache_dir is None:
        if 'SM_CHANNEL_TRAINING' in os.environ:
            cache_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'cache')
        else:
            cache_dir = '/tmp/huggingface_cache'

    os.makedirs(cache_dir, exist_ok=True)

    # Use torchvision FashionMNIST directly
    try:
        train_ds = datasets.FashionMNIST(root=cache_dir, train=True, download=True, transform=None)
        test_ds = datasets.FashionMNIST(root=cache_dir, train=False, download=True, transform=None)
    except Exception as e:
        print(f"Download warning: {e}")
        train_ds = datasets.FashionMNIST(root=cache_dir, train=True, download=False, transform=None)
        test_ds = datasets.FashionMNIST(root=cache_dir, train=False, download=False, transform=None)

    # Use our standard TorchVisionWrapper
    train_ds_wrapped = TorchVisionWrapper(train_ds).with_transform(fmnist_transform)
    test_ds_wrapped = TorchVisionWrapper(test_ds).with_transform(fmnist_transform)

    return train_ds_wrapped, test_ds_wrapped, 10, 1  # 10 classes, 1 channel


def get_dataset(name):
    name = name.lower()
    if name == "mnist":
        return get_mnist("/tmp/huggingface_cache")
    elif name == "cifar10":
        return get_cifar10()
    elif name == "cifar100":
        return get_cifar100()
    elif name == "pathmnist":
        return get_pathmnist()
    elif name == "eurosat":
        return get_eurosat()
    elif name == "fashionmnist":
        return get_fashionmnist()
    raise ValueError(f"Dataset {name} not implemented in this demo.")

# ---------------------------
# Non-IID partition (Dirichlet)
# ---------------------------
def extract_labels_plain(ds, label="label"):
    """Extracts labels of the dataset as a Python list."""
    ds_plain = ds.with_format(type=None)
    return [int(y) for y in ds_plain[label]]

def _to_poisoned_id_set(poisoned: Union[Dict[int, object], Iterable[int], None]) -> set:
    if poisoned is None:
        return set()
    if isinstance(poisoned, dict):
        return set(poisoned.keys())
    return set(poisoned)

def dirichlet_partition(
    labels: List[int],
    poisoned: Union[Dict[int, object], Iterable[int], None],
    num_clients: int,
    alpha: float,
    test_size: float,
    poison_size: float = None
) -> Dict[int, Dict[str, List[int]]]:
    """Return per-client indices: {'all','train','test','poisoned'}."""
    assert 0 < test_size < 1, "test_size must be in (0,1)"
    if poison_size is not None:
        assert 0 < poison_size < 1, "poison_size must be in (0,1)"

    poisoned_ids = _to_poisoned_id_set(poisoned)

    labels_t = torch.tensor(labels)
    classes = labels_t.unique().tolist()
    idx_by_c = {c: (labels_t == c).nonzero(as_tuple=True)[0].tolist() for c in classes}
    for c in idx_by_c: random.shuffle(idx_by_c[c])

    client_indices = {client: {"all": [], "train": [], "test": [], "poisoned": []}
                      for client in range(num_clients)}

    # allocate per-class via Dirichlet
    for c, idxs in idx_by_c.items():
        props = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample().tolist()
        splits = [int(p * len(idxs)) for p in props]
        while sum(splits) < len(idxs): splits[random.randrange(num_clients)] += 1
        while sum(splits) > len(idxs):
            for j in range(num_clients):
                if splits[j] > 0 and sum(splits) > len(idxs): splits[j] -= 1
        start = 0
        for j, sz in enumerate(splits):
            if sz > 0:
                client_indices[j]["all"].extend(idxs[start:start+sz])
                start += sz

    # per-client splits
    for j in range(num_clients):
        random.shuffle(client_indices[j]["all"])
        n_all = len(client_indices[j]["all"])
        n_test = int(math.floor(n_all * test_size))
        client_indices[j]["test"] = client_indices[j]["all"][:n_test]
        client_indices[j]["train"] = client_indices[j]["all"][n_test:]
        if (poison_size is not None) and (j in poisoned_ids):
            n_poison = int(math.floor(len(client_indices[j]["train"]) * poison_size))
            client_indices[j]["poisoned"] = client_indices[j]["train"][:n_poison]

    return client_indices