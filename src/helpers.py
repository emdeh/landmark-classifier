from io import BytesIO
import urllib.request
from zipfile import ZipFile
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing

import numpy as np
import random
import matplotlib.pyplot as plt


def setup_env():
    """
    1. Detect whether CUDA is available. If not, enumerate all DirectML adapters,
       pick the one named 'RX 7900 XT' (if present), otherwise use the default.
       Fall back to CPU only if no DML adapter is usable.
    2. Seed RNGs (torch.manual_seed always; torch.cuda.manual_seed_all if using CUDA).
    3. Download & extract data if needed, then compute mean/std.
    4. Create 'checkpoints' folder if missing.
    5. Adjust PATH for Udacity workspace if necessary.
    Returns the chosen device (torch.device('cuda'), a torch_directml device, or torch.device('cpu')).
    """
    # -------- 1. Device selection --------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        # No CUDA → try DirectML
        try:
            import torch_directml

            n = torch_directml.device_count()
            print(f"Found {n} DirectML adapter(s):")
            for idx in range(n):
                name = torch_directml.device_name(idx)
                print(f"  • DML device-{idx} name: {name!r}")

            # Look for "RX 7900 XT" in adapter names
            chosen_idx = None
            for idx in range(n):
                if "RX 7900 XT" in torch_directml.device_name(idx):
                    chosen_idx = idx
                    break

            if chosen_idx is None:
                # If we didn’t find “RX 7900 XT”, fall back to default_device()
                chosen_idx = torch_directml.default_device()
                chosen_name = torch_directml.device_name(chosen_idx)
                print(f"No adapter explicitly named 'RX 7900 XT' found. Using default index {chosen_idx}: {chosen_name!r}")
            else:
                chosen_name = torch_directml.device_name(chosen_idx)
                print(f"Selecting adapter #{chosen_idx}: {chosen_name!r} (RX 7900 XT)")

            # Create the DirectML device object for that index
            device = torch_directml.device(chosen_idx)

        except Exception as e:
            # Any failure at all → fallback to CPU
            device = torch.device("cpu")
            print("No CUDA or usable DirectML adapters found. Using CPU (slow).")
            # Optionally print the exception for debugging:
            # print("  Error:", e)

    # -------- 2. Seed RNGs ----------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -------- 3. Download & compute mean/std ----------
    download_and_extract()
    compute_mean_and_std()

    # -------- 4. Create checkpoints dir ----------
    os.makedirs("checkpoints", exist_ok=True)

    # -------- 5. Adjust PATH for Udacity workspace ----------
    if os.path.exists("/data/DLND/C2/landmark_images"):
        os.environ["PATH"] = f"{os.environ['PATH']}:/root/.local/bin"

    return device




def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    elif os.path.exists("/data/DLND/C2/landmark_images"):
        data_folder = "/data/DLND/C2/landmark_images"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    
    try:
        
        location = get_data_location()
    
    except IOError:
        # Dataset does not exist
        print(f"Downloading and unzipping {url}. This will take a while...")

        with urllib.request.urlopen(url) as resp:

            with ZipFile(BytesIO(resp.read())) as fp:

                fp.extractall(".")

        print("done")
                
    else:
        
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )
        return None


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)
