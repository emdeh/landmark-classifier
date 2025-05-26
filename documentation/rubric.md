# Project Rubric
***[Back home](/README.md)***

## ✅ File Requirements
- All required files are submitted:
  - Notebooks: `cnn_from_scratch.ipynb`, `transfer_learning.ipynb`, `app.ipynb`
  - Python scripts in `src/`: `train.py`, `model.py`, `helpers.py`, `predictor.py`, `transfer.py`, `optimization.py`, `data.py`

---

## 🧠 CNN from Scratch (cnn_from_scratch.ipynb)

### Data Preparation (`src/data.py`)
- All `YOUR CODE HERE` sections completed
- `data_transforms` includes:
  - Resize(256)
  - Crop (RandomCrop for train, CenterCrop for valid/test)
  - ToTensor
  - Normalize (with dataset mean and std)
  - Augmentation for training (e.g., flip, rotate)
- Uses appropriate `ImageFolder` instances
- Data loaders correctly implemented (batch_size, sampler, num_workers)
- All tests in notebook pass

### Data Visualisation
- `visualize_one_batch` shows 5 images from the train loader with labels

### Model Definition (`src/model.py`)
- `MyModel` class correctly implements CNN
- Uses `num_classes` parameter (not hardcoded)
- Dropout controlled via constructor
- No softmax in forward pass
- All model-related tests pass

### Model Architecture Description
- Explanation includes:
  - Chosen architecture
  - Reasoning behind layer types, order, etc.
  - Concepts reused from coursework

### Loss & Optimiser (`src/optimization.py`)
- `get_loss()` returns CrossEntropyLoss
- `get_optimizer()` implements both SGD and Adam
- All optimiser/loss tests pass

### Training and Validation (`src/train.py`)
- `train_one_epoch()` and `valid_one_epoch()` correctly implemented
- Training: model.train(), loss backprop, optimiser.step()
- Validation: model.eval(), forward only
- `optimize()`:
  - Scheduler with LR reduction on plateau
  - Checkpointing on ≥1% improvement
- `one_epoch_test()` evaluates and returns predictions
- All tests pass

### Full Pipeline Integration
- All components are correctly wired
- Reasonable hyperparameters
- Training converges (loss decreases)
- Test accuracy ≥ 50%

### TorchScript Export (`src/predictor.py`)
- `forward()`:
  - Applies self.transforms
  - Applies model
  - Applies softmax across dim=1
- Exports with `torch.jit.script`
- Reloads and validates with confusion matrix (visible diagonal)

---

## 🔁 Transfer Learning (transfer_learning.ipynb)

### Architecture (`src/transfer.py`)
- Pretrained model loaded
- Backbone frozen
- Final `nn.Linear` layer added with appropriate input/output features
- All Step 1 tests pass

### Training
- Sensible hyperparameters
- Uses `get_model_transfer_learning()`
- Convergence visible (train and val loss reduce)

### Architecture Justification
- Explanation provided for model suitability

### Testing
- Test accuracy ≥ 60%

### Export
- Model exported as `checkpoints/transfer_exported.pt`

---

## 📱 App Deployment (app.ipynb)

### App Functionality
- All `YOUR CODE HERE` sections completed
- Loads one TorchScript model via `torch.jit.load`
- App runs inference on a new (unseen) image
- Displays image and top predictions

___

***[Back home](/README.md)***