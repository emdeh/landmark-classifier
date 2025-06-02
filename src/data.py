import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


# --- Wrapper so that iter(loader).next() works --- 
class DataLoaderWithNext:
    """
    Wrap a DataLoader so that its iterator has a .next() method aliasing __next__(),
    allowing calls to dataiter.next() to succeed.
    """
    def __init__(self, dataloader):
        self.dl = dataloader

    def __iter__(self):
        it = iter(self.dl)
        it.next = it.__next__
        return it

    def __len__(self):
        return len(self.dl)


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, pin_memory: bool = True, limit: int = -1
):
    """
    Create and return train, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset for validation (e.g. 0.2)
    :param num_workers: number of worker processes; -1 means “all cores”
    :param limit: maximum number of data points to consider (for debugging)
    :return: dict with keys 'train', 'valid' and 'test' mapping to DataLoaderWithNext
    """
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())

    # Compute mean and std
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Define transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    # Load train data (for split into train/valid)
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"],
    )
    valid_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["valid"],
    )

    # Create indices for train/valid split
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # Wrap DataLoader so .next() exists
    data_loaders["train"] = DataLoaderWithNext(
        torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )
    data_loaders["valid"] = DataLoaderWithNext(
        torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )

    # Load test data
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform=data_transforms["test"],
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = DataLoaderWithNext(
        torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualise one batch of data.
    """
    train_loader = data_loaders["train"]
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )
    images = invTrans(images)

    # Access the underlying dataset to get class names
    class_names = train_loader.dl.dataset.classes

    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid", "test"}, \
        "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, \
        "The tensors returned by your dataloaders should be 224x224. Did you forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert len(labels) == 2, f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
