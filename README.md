# OCR on CAPTCHA Using DenseNet

A deep learning project for **CAPTCHA OCR** using a **DenseNet121** backbone, built with **PyTorch Lightning** and tracked with **MLflow**.

Repository: https://github.com/Xenox-cloud/OCR_on_captcha_using_densenet

This project is inspired by the paper **“CAPTCHA recognition based on deep convolutional neural network”** by Wang et al. (2019), which uses a DenseNet-style architecture to improve OCR quality on distorted CAPTCHA images.

## Overview

Traditional OCR systems struggle with CAPTCHA images because the text is intentionally distorted, noisy, and hard to segment. This repository approaches the problem as an end-to-end image recognition task:

- CAPTCHA images are loaded directly from disk
- labels are extracted from the image filenames
- a **DenseNet121** model learns visual features from grayscale images
- the network predicts the 4-character CAPTCHA string
- training and validation are tracked with MLflow

The current implementation uses a **fixed-length multi-character classifier** rather than explicit character segmentation.

## Features

- **DenseNet121** backbone for stronger visual feature extraction
- grayscale image support
- fixed-length CAPTCHA prediction (`MAX_CAPTCHA = 4`)
- PyTorch Lightning training pipeline
- MLflow experiment tracking
- checkpointing and early stopping
- OCR-oriented evaluation metrics beyond loss alone

## Repository Structure

```text
OCR_on_captcha_using_densenet/
├── data/
│   └── OCR/
│       ├── Train/
│       └── Test/
├── notebook.ipynb
├── notebook.py
├── requirements.txt
└── README.md
```

## Dataset Format

The label is taken directly from the file name.

Example:

```text
data/OCR/Train/0b3b.png
```

The target label for that image is:

```text
0b3b
```

The current character set includes:

- digits: `0-9`
- lowercase letters: `a-z`

Total classes per position: **36**

## Model

The project uses **DenseNet121** from `torchvision.models` as the feature extractor.

### Architecture

```text
Input CAPTCHA Image
        │
        ▼
DenseNet121 Backbone
        │
        ▼
Fully Connected Classifier
        │
        ▼
4 Character Predictions
```

### Why DenseNet?

DenseNet is a strong fit for CAPTCHA OCR because it:

- improves gradient flow through dense connections
- reuses features across layers
- preserves fine visual details
- handles noisy and distorted character patterns better than simpler CNNs

## Metrics

This project tracks several metrics during training and validation.

### 1. Character Accuracy
Measures how many individual characters are predicted correctly.

### 2. CAPTCHA Accuracy
Measures whether the **entire CAPTCHA** is predicted correctly.

This is the most important metric for the task.

### 3. Average Edit Distance
Measures how many character-level edits are needed to transform the prediction into the correct label.

### 4. Average Normalized Edit Distance
Edit distance normalized by target string length.

### 5. Confidence
Average softmax confidence of the model’s predictions.

### 6. Per-Position Accuracy
Tracks accuracy for each CAPTCHA position separately.

## Training Setup

The training script currently uses:

- **Optimizer:** AdamW
- **Scheduler:** ReduceLROnPlateau
- **Checkpoint monitor:** `val_captcha_acc`
- **Early stopping:** based on `val_captcha_acc`
- **Batch size:** 32
- **Image size:** `64 x 128`

## MLflow Tracking

MLflow is integrated to log:

- hyperparameters
- training and validation metrics
- best checkpoint path
- saved model artifacts

### Run MLflow UI locally

```bash
mlflow ui --port 5050
```

Then open:

```text
http://127.0.0.1:5050
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Xenox-cloud/OCR_on_captcha_using_densenet.git
cd OCR_on_captcha_using_densenet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Python script

```bash
python notebook.py
```

### Or use the notebook

Open:

```text
notebook.ipynb
```

## Example Inference

The script includes a simple prediction example:

```python
image_file = r".\data\OCR\Test\nfj8.png"
pred_text = model.predict_image(image_file)
print(pred_text)
```

## Current Pipeline

```text
CAPTCHA Image
      │
      ▼
Preprocessing (resize + grayscale)
      │
      ▼
DenseNet121
      │
      ▼
4-position character classifier
      │
      ▼
Predicted CAPTCHA text
```

## Inspiration

This project was improved using ideas from:

**Wang, J., Qin, J., Xiang, X., Tan, Y., & Pan, N. (2019).**  
*CAPTCHA recognition based on deep convolutional neural network.*  
Mathematical Biosciences and Engineering.  
Paper link: https://www.aimspress.com/article/10.3934/mbe.2019292

## Notes

- The current implementation uses the `Test` folder as the validation set.
- Labels are assumed to have maximum length 4.
- Images are loaded in grayscale mode.
- The model is saved as `captcha_model.pth` after training.

## Possible Future Improvements

- create a separate validation split instead of using the test set directly
- add stronger data augmentation for CAPTCHA distortions
- compare DenseNet with ResNet and CRNN-based OCR models
- export the trained model for deployment via API or web app
- add a results section with sample predictions and benchmark numbers
