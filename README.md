# Optical Character Recognition Using PyTorch

This project implements an Optical Character Recognition (OCR) model using PyTorch. The model is designed to recognize individual characters in natural images by training a Convolutional Neural Network (CNN).

## Table of Contents

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Data Loading and Preparation](#data-loading-and-preparation)
- [Building the CNN](#building-the-cnn)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)

## Dependencies

```bash
pip install torch torchvision matplotlib
```

## Dataset

Chars74k

## Image Preprocessing

Images are preprocessed before being fed into the model:

1. Resize images to 48x48 pixels.
2. Randomly flip images horizontally for data augmentation.
3. Convert images into PyTorch tensors and normalize pixel values.

```python
data = torchvision.datasets.ImageFolder(
    root='./EnglishFnt/English/Fnt',
    transform=transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
```

## Data Loading and Preparation

The dataset is split into training, validation, and test sets. A data loader is created for each set.

```python
def load_split(dataset, batch_size, test_split=0.3, random_seed=42):
    # Code for loading and splitting data
```

## Building the CNN

We define a simple CNN architecture for the OCR model:

```python
class OCRNet(nn.Module):
    def __init__(self, num_features):
        super(OCRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(num_features, 62)  # Output has 62 classes instead of 10
```

## Training the Model

The model is trained using Stochastic Gradient Descent (SGD) and Cross Entropy Loss.

```python
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

## Making Predictions

The trained model can be used to predict the class of a single image.

```python
def predict(model, image_path, transform):
    # Code for making predictions
```

