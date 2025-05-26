# Landmark Classifier
A project as part of a Convolutional Neural Network (CNN) course to build a landmark classifier.

Photo sharing and storage platforms benefit from having location data associated with uploaded images, as it enables features like automatic tagging and intelligent photo organisation. While this data is often available through photo metadata, it’s not always present. This may be due to cameras lacking GPS capability or metadata being removed for privacy reasons.

When location metadata is unavailable, one way to infer a photo’s origin is by identifying and classifying any visible landmarks. Given the global scale and volume of images uploaded daily, manual classification isn't practical.

In this project, we develop a model that predicts the location of an image based on landmarks it contains. This includes:
-  preprocessing data,
- designing and training CNNs,
- evaluating model performance, and
- deploying an application using the best-performing model.

## Documenation
- [Rubric](/documentation/rubric.md)
- [Environment setup](/documentation/setup.md)
- [Checklist](/documentation/checklist.md)

# Project steps
The project is structured into three key phases:

## 1. Build a CNN from Scratch to Classify Landmarks
Dataset exploration and visualisation, and data preparation for training and constructing a CNN to classify landmarkes. As part of this phase, we document key decisions, such as preprocessing steps and how network architecture is designed. Once trained, we export the best-performing model using TorchScript.

## 2. Apply Transfer Learning to Classify Landmarks
In this phase, we evaluate several pre-trained models and select one to adapt for the classification task. We train and test this transfer learning model and explain the reasoning behind the chosen architecture. The top-performing model will also be exported using TorchScript.

## 3. Deploy the Model in an App
Finally, we build a simple application that allows users to input an image and receive predictions for likely landmarks. We test the app, evaluate the model’s performance in practice, and reflect on its strengths and limitations.

# Project evaluation

See the [Project Rubric](/documentation/rubric.md) for details on the project evaluation.

## 1. File requirements
- All specified files must be included: 3 notebooks (`cnn_from_scratch.ipynb`, `transfer_learning.ipynb`, `app.ipynb`) and 7 Python scripts under `src/`.

## 2. CNN from scratch (`cnn_from_scratch.ipynb`)
- Data Pipeline (`data.py`): Preprocessing (resize, crop, normalisation, augmentation), proper data loader setup, and passing automated tests
- Visualisation: Display sample images with labels.
- Model Definition (`model.py`): Custom CNN with `num_classes` support, no hardcoded softmax, optional dropout.
- Optimisation (`optimization.py`): Correct loss (CrossEntropy), correct optimiser config (SGD or Adam), all tests pass.
- Training/Validation (`train.py`): Training and validation loop correctness, scheduler and model checkpointing, final evaluation on test set.
- TorchScript Export (`predictor.py`): Implements forward pass with softmax, model is exported and validated with confusion matrix.

## 3. Transfer Learning (`transfer_learning.ipynb`)
- Model Setup (`transfer.py`): Freezing pretrained backbone, appending correct final layer, all tests pass.
- Training: Performance trends as expected, test accuracy ≥ 60%.
- Export: Model is TorchScript-exported properly.

## 4. App Deployment (`app.ipynb`)
- A working app loads the exported model, runs inference on a new image, and displays results.

# Other considerations

Continue refining the model and tuning training parameters to maximise test accuracy.

- For the **CNN from scratch**, aim for **>60%** test accuracy.
- For the **transfer learning model**, a well-chosen architecture can reach **≥80%**.

### 🔬 Suggested Areas to Experiment With
- **Data augmentation** techniques (e.g. flipping, rotating, colour jitter)
- **Different model architectures** to test depth and complexity
- **Regularisation methods** like increased dropout or weight decay
- **Batch size** variations to observe effects on stability and convergence

### Track Experiments
Maintain a structured log of each experiment, including:
- Training loss
- Validation loss
- Validation accuracy
- Observations (e.g. _"Model overfitting: validation loss plateaued while training loss continued to drop."_)

Indicate clearly which setup achieved the best validation performance.

> **Important:** Use only the **validation set** to tune hyperparameters. After finalising the model, evaluate it on the **test set** once to confirm generalisation.

---

## 🧠 Bonus Task: Image Retrieval System

Use the **penultimate layer features** of the best CNN to implement a basic image retrieval system:

1. Extract the feature vector for a query image.
2. Compute dot products between the query vector and feature vectors of images in the `landmark_images` dataset.
3. Return the images with the highest similarity scores.

---

## 🌍 Broader Applications

Include a short discussion of **other potential use cases** for your model or feature-based retrieval system. For example:

- Smart photo organisation or recommendation systems
- Landmark detection in tourism and travel apps
- Cultural heritage site monitoring
- Visual search for educational tools
- Environmental change detection from imagery

Show how your work could extend beyond this specific task to solve real-world problems.
