# Optical Character Recognition Using PyTorch

This project implements an Optical Character Recognition (OCR) model using PyTorch. The model is designed to recognize individual characters in natural images by training a Convolutional Neural Network (CNN).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Data Loading and Preparation](#data-loading-and-preparation)
- [Building the CNN](#building-the-cnn)
- [Training the Model](#training-the-model)
- [Validation and Testing](#validation-and-testing)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [License](#license)

## Installation

To get started, make sure you have the following packages installed:

```bash
pip install torch torchvision matplotlib
```

If you're using Google Colab, you can directly run the provided code without any additional installation.

## Dataset

We utilize the **Chars74K** dataset, which contains images of individual characters with various scales. We specifically use the `EnglishFnt.tgz` file, which includes:

- **62 Classes:** Digits (0-9), uppercase letters (A-Z), and lowercase letters (a-z).
- **Variations:** Characters from computer fonts with four styles (italic, bold, and normal).

### Downloading the Dataset

The dataset can be downloaded directly or accessed via Google Drive if mounted.

```python
import tarfile

# For Google Drive users
if drive_mounted:
    tar = tarfile.open('/content/drive/MyDrive/Python_OCR/EnglishFnt.tgz')
else:
    !wget -O EnglishFnt.tgz http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
    tar = tarfile.open('EnglishFnt.tgz')

tar.extractall('./EnglishFnt')
tar.close()
```

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

## Validation and Testing

After training, the model is validated and tested using the respective datasets.

```python
def validate(model, val_loader, criterion):
    # Code for validation
```

```python
def test(model, test_loader, criterion):
    # Code for testing
```

## Saving and Loading the Model

You can save the trained model for later use or further training.

```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

## Making Predictions

The trained model can be used to predict the class of a single image.

```python
def predict(model, image_path, transform):
    # Code for making predictions
```

## Usage

1. Ensure you have the dataset ready.
2. Follow the code snippets in each section to train and validate your OCR model.
3. Use the `predict` function to test the model with new images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
