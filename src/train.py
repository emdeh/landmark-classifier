import tempfile
<<<<<<< HEAD
=======

>>>>>>> origin/main
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


<<<<<<< HEAD
def _get_default_device():
    """
    Additional helper to select a default device if none is provided.
    Preference: CUDA > DirectML > CPU.
    Needed this because of my AMD GPU setup which uses DirectML.
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

=======
def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        # YOUR CODE HERE: transfer the model to the GPU
        # HINT: use .cuda()

    # YOUR CODE HERE: set the model to training mode
    
>>>>>>> origin/main
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
<<<<<<< HEAD
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_value = loss_fn(output, target)

        # ─── On the first batch of the epoch, print loss, accuracy, and gradient norm ───
        if batch_idx == 0:
            preds = output.argmax(dim=1)
            batch_acc = (preds == target).float().mean().item()
            print(f"[Batch 0] pre-step loss = {loss_value.item():.4f}, acc = {100*batch_acc:.1f}%")

        loss_value.backward()

        if batch_idx == 0:
            total_grad = sum(
                p.grad.detach().abs().sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            print(f"[Batch 0] ∑|grad| = {total_grad:.2e}")
        # ───────────────────────────────────────────────────────────────────────────────────────

        optimizer.step()

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.item() - train_loss)
=======
        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        # YOUR CODE HERE:
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = # YOUR CODE HERE
        # 3. calculate the loss
        loss_value  = # YOUR CODE HERE
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        # YOUR CODE HERE:
        # 5. perform a single optimization step (parameter update)
        # YOUR CODE HERE:

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
>>>>>>> origin/main
        )

    return train_loss


<<<<<<< HEAD


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
=======
def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # set the model to evaluation mode
        # YOUR CODE HERE

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
>>>>>>> origin/main
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
<<<<<<< HEAD
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss_fn(output, target)
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.item() - valid_loss)
=======
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = # YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = # YOUR CODE HERE

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
>>>>>>> origin/main
            )

    return valid_loss


<<<<<<< HEAD
def optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, device=None, interactive_tracking=False, scheduler_patience = 10):
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
    print("Starting optimization...")
    if device is None:
        device = _get_default_device()

    model = model.to(device)
    print("Optimising on:", next(model.parameters()).device)

=======
def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
>>>>>>> origin/main
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}
<<<<<<< HEAD
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
        threshold=1e-2,
        verbose=True,
        min_lr=1e-6,  # Prevents LR from going too low
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  → LR after epoch {epoch}: {current_lr:.6f}")
        logs["lr"] = current_lr

=======

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here: 
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler  = # YOUR CODE HERE

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            # YOUR CODE HERE

            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        # YOUR CODE HERE

        # Log the losses and the current learning rate
>>>>>>> origin/main
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
<<<<<<< HEAD
=======

>>>>>>> origin/main
            liveloss.update(logs)
            liveloss.send()


<<<<<<< HEAD
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
=======
def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        # YOUR CODE HERE

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = # YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = # YOUR CODE HERE

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred  = # YOUR CODE HERE

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
>>>>>>> origin/main

    return test_loss


<<<<<<< HEAD
=======
    
>>>>>>> origin/main
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
<<<<<<< HEAD
    from src.data import get_data_loaders
=======
    from .data import get_data_loaders
>>>>>>> origin/main

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)
<<<<<<< HEAD
=======

>>>>>>> origin/main
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
<<<<<<< HEAD
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders["train"], model, optimizer, loss)
=======

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
>>>>>>> origin/main
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
<<<<<<< HEAD
    model, loss, _ = optim_objects
=======

    model, loss, optimizer = optim_objects

>>>>>>> origin/main
    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

<<<<<<< HEAD

def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
=======
def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

>>>>>>> origin/main
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
<<<<<<< HEAD
    model, loss, _ = optim_objects
=======

    model, loss, optimizer = optim_objects

>>>>>>> origin/main
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
