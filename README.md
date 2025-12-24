
# CIFAR-10 Image Classification using MobileNetV2 and Transfer Learning (PyTorch)

## Project Overview

This project demonstrates an efficient image classification pipeline using **MobileNetV2** with **transfer learning** on the **CIFAR-10 dataset**.
The goal is to build a low-compute, high-accuracy image classifier suitable for resource-constrained environments.

The model is trained in two phases:

1. Feature extraction with frozen backbone
2. Fine-tuning the last convolutional layers for improved accuracy

## Dataset

**CIFAR-10** is a standard computer vision dataset consisting of **60,000 RGB images (32×32)** across **10 classes**.

### Classes

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

## Technologies Used

* Python 3
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook
* MobileNetV2 (pretrained on ImageNet)

## Model Architecture

* Backbone: MobileNetV2 (pretrained)
* Input size: 224 × 224
* Classifier: Fully connected layer replaced for 10 classes
* Loss function: CrossEntropyLoss
* Optimizer: Adam
* Training strategy:

  * Phase 1: Freeze backbone parameters
  * Phase 2: Unfreeze last convolution block (layer4) for fine-tuning

## Data Preprocessing

### Training Transforms

* Resize to 224×224
* Random horizontal flip
* Random rotation
* Convert to tensor
* Normalize using ImageNet mean and std

### Testing Transforms

* Resize to 224×224
* Convert to tensor
* Normalize using ImageNet mean and std

## Training Details

* Batch size: 64
* Device: GPU (CUDA) if available, else CPU
* Initial learning rate: 0.001
* Fine-tuning learning rate: 0.0001
* Epochs:

  * Phase 1: 5 epochs
  * Phase 2: 10 epochs

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Evaluation is performed using `sklearn.metrics.classification_report` and confusion matrix visualization with Matplotlib.

## Results Summary

* Final test accuracy: ~83%
* Strong performance across most classes
* Confusion matrix shows clear diagonal dominance, indicating correct predictions
* Some confusion observed between visually similar classes (cat/dog, automobile/truck)

## Project Structure

```
project-root/
│
├── README.md
├── notebook.ipynb
├── data/
│   └── cifar-10-batches-py/
└── outputs/
    └── confusion_matrix.png
```

## Installation

Create a virtual environment (recommended) and install dependencies.

```
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

For GPU support, install PyTorch according to your CUDA version from the official PyTorch website.

## How to Run the Project

1. Clone the repository

```
git clone <your-repository-url>
cd <project-folder>
```

2. Start Jupyter Notebook

```
jupyter notebook
```

3. Open the notebook file

```
notebook.ipynb
```

4. Run all cells in sequence

   * Dataset will download automatically
   * Model will train and fine-tune
   * Evaluation metrics and confusion matrix will be displayed

## Use Cases

* Learning transfer learning with PyTorch
* Low-compute image classification
* Academic projects and assignments
* Resume and portfolio projects
* Interview demonstration of deep learning concepts

## Future Improvements

* Add learning rate scheduler
* Experiment with data augmentation techniques
* Try other lightweight models (EfficientNet, ShuffleNet)
* Export model for deployment (ONNX / TorchScript)
* Add inference script for single image prediction


* Write a **LinkedIn project post**
* Create a **project report PDF**
* Add an **inference script** for single image testing

Just tell me.
