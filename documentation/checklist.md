# ✅ Landmark Classifier Project Checklist


***[Back home](/README.md)***

Track your progress as you complete each phase of the project. Each task corresponds to specific rubric items and deliverables.

---

## 📁 File Structure & Submission

- [ ] Ensure the following notebooks exist and are complete:
  - [ ] `cnn_from_scratch.ipynb`
  - [ ] `transfer_learning.ipynb`
  - [ ] `app.ipynb`
- [ ] Ensure the following Python files are complete under `/src`:
  - [ ] `data.py`
  - [ ] `model.py`
  - [ ] `train.py`
  - [ ] `optimization.py`
  - [ ] `predictor.py`
  - [ ] `transfer.py`
  - [ ] `helpers.py`
- [ ] Add documentation files:
  - [ ] `README.md`
  - [ ] `rubric.md`
  - [ ] `checklist.md`
  - [ ] `setup.md` (for environment setup instructions)

---

## 🧠 Phase 1: CNN from Scratch

### Data Preparation (`src/data.py`)
- [ ] Implement `get_data_loaders()` with correct transforms (Resize, Crop, ToTensor, Normalize, Augmentations)
- [ ] Implement `visualize_one_batch()` to display 5 training images with labels
- [ ] Pass all data loader tests in notebook

### Model Definition (`src/model.py`)
- [ ] Implement `MyModel` class using configurable `num_classes` and `dropout`
- [ ] Omit softmax in `forward()` method
- [ ] Pass all model tests in notebook

### Optimisation (`src/optimization.py`)
- [ ] Implement `get_loss()` with `CrossEntropyLoss`
- [ ] Implement `get_optimizer()` supporting SGD and Adam
- [ ] Pass all optimisation tests

### Training (`src/train.py`)
- [ ] Implement `train_one_epoch()`, `valid_one_epoch()`, and `optimize()` correctly
- [ ] Implement `one_epoch_test()` for evaluation
- [ ] Model trains with converging loss
- [ ] Test accuracy ≥ 50%

### TorchScript Export (`src/predictor.py`)
- [ ] Implement `forward()` with transforms and softmax
- [ ] Save best model using `torch.jit.script`
- [ ] Reload and validate with confusion matrix

---

## 🔁 Phase 2: Transfer Learning

### Model Setup (`src/transfer.py`)
- [ ] Load and freeze pretrained model
- [ ] Append correct final `nn.Linear` head
- [ ] Pass all model setup tests

### Training & Evaluation
- [ ] Train transfer learning model with proper hyperparameters
- [ ] Monitor loss convergence
- [ ] Achieve test accuracy ≥ 60%

### Export
- [ ] Save final transfer model to `checkpoints/transfer_exported.pt`

---

## 📱 Phase 3: App Deployment

- [ ] Load one of the TorchScript models using `torch.jit.load`
- [ ] Run inference on an unseen image
- [ ] Display input image and predicted landmark(s)
- [ ] Complete all `YOUR CODE HERE` sections in `app.ipynb`

---

## 🧪 Experimentation & Tuning

- [ ] Conduct multiple training experiments with varied:
  - [ ] Data augmentations
  - [ ] Architectures
  - [ ] Dropout/regularisation values
  - [ ] Batch sizes
- [ ] Track results in a table:
  - [ ] Training loss
  - [ ] Validation loss
  - [ ] Validation accuracy
  - [ ] Observations per run
- [ ] Identify best setup (without using test set)

---

## 🧠 Bonus: Image Retrieval System

- [ ] Extract penultimate-layer features from CNN
- [ ] Compute dot product similarity with `landmark_images` dataset
- [ ] Return most similar images
- [ ] Include brief implementation notes or code demo

---

## 🌍 Broader Applications

- [ ] Write a short discussion of other potential use cases (e.g. smart albums, tourism, heritage monitoring)

---

✅ **Final Checks**
- [ ] All notebooks run top-to-bottom without error
- [ ] All `YOUR CODE HERE` sections are filled
- [ ] All tests pass in notebooks
- [ ] Submission includes only required and working files

---

> Tip: Use `[x]` to check off completed items as you go.
___

***[Back home](/README.md)***