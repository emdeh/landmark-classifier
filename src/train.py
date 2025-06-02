import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def _get_default_device():
    """
    Helper to select a default device if none is provided.
    Preference: CUDA > DirectML > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml

        n = torch_directml.device_count()
        if n > 0:
            # Attempt to pick an AMD RX 7900 XT if present
            chosen_idx = None
            for idx in range(n):
                name = torch_directml.device_name(idx)
                if "RX 7900 XT" in name:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                chosen_idx = torch_directml.default_device()
            return torch_directml.device(chosen_idx)
    except Exception:
        pass

    return torch.device("cpu")


def train_one_epoch(train_dataloader, model, optimizer, loss_fn, device=None):
    """
    Performs one training epoch.

    Args:
      train_dataloader: DataLoader for training data.
      model:            nn.Module.
      optimizer:        torch.optim.Optimizer.
      loss_fn:          loss function (e.g. nn.CrossEntropyLoss()).
      device:           torch.device or DirectML device. If None, auto-select.

    Returns:
      train_loss (float): average loss over all batches.
    """
    if device is None:
        device = _get_default_device()

    model = model.to(device)
    model.train()
    print("training on:", next(model.parameters()).device)

    train_loss = 0.0
    
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_value = loss_fn(output, target)
        loss_value.backward()
        
        ## ───── DEBUG: print total grad magnitude (first batch each epoch) ─────
        #if batch_idx == 0:                        # only once per epoch
        #    total_grad = sum(
        #        p.grad.abs().sum().item()
        #        for p in model.parameters()
        #        if p.grad is not None
        #    )
        #    print(f"∑|grad| = {total_grad:.2e}")
        ## ──────────────────────────────────────────────────────────────────────

        optimizer.step()

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.item() - train_loss)
        )

    

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss_fn, device=None):
    """
    Validate at the end of one epoch.

    Args:
      valid_dataloader: DataLoader for validation data.
      model:            nn.Module.
      loss_fn:          loss function.
      device:           torch.device or DirectML device. If None, auto-select.

    Returns:
      valid_loss (float): average loss over validation set.
    """
    if device is None:
            device = _get_default_device()

    model = model.to(device)
    model.eval()
    print("Validating on:", next(model.parameters()).device)

    valid_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss_fn(output, target)
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, device=None, interactive_tracking=False, scheduler_patience: int = 20):
    """
    Full training + validation loop over n_epochs. Saves model when validation loss improves by ≥1%.

    Args:
      data_loaders:         dict with keys "train" and "valid" mapping to DataLoader.
      model:                nn.Module.
      optimizer:            torch.optim.Optimizer.
      loss_fn:              loss function.
      n_epochs:             number of epochs.
      save_path:            path to save best model state_dict.
      device:               torch.device or DirectML device. If None, auto-select.
      interactive_tracking: whether to use livelossplot for real-time charts.
    """
    if device is None:
        device = _get_default_device()

    model = model.to(device)
    print("Optimising on:", next(model.parameters()).device)

    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=scheduler_patience,
        threshold=1e-4

    )

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss_fn, device)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss_fn, device)

        print(
            f"Epoch: {epoch} \t"
            f"Training Loss: {train_loss:.6f} \t"
            f"Validation Loss: {valid_loss:.6f}"
        )

        if valid_loss_min is None or (valid_loss_min - valid_loss) / valid_loss_min > 0.01:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model …")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        scheduler.step(valid_loss)

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss_fn, device=None):
    """
    Run one full test epoch, reporting loss + accuracy.

    Args:
      test_dataloader: DataLoader for test data.
      model:           nn.Module.
      loss_fn:         loss function.
      device:          torch.device or DirectML device. If None, auto-select.

    Returns:
      test_loss (float): average loss over test set.
    """
    if device is None:
        device = _get_default_device()
        
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc="Testing",
            total=len(test_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss_value = loss_fn(logits, target)

            test_loss = test_loss + (
                (1 / (batch_idx + 1)) * (loss_value.item() - test_loss)
            )

            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds.eq(target)).item()
            total += data.size(0)

    print(f"Test Loss:  {test_loss:.6f}")
    print(f"Test Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

    return test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from src.data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, _ = optim_objects
    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, _ = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
