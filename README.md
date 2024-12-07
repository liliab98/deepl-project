# Medical Image Classification Using a Tailored ResNet50 Model

This repository contains the code for the mandatory project in the **Deep Learning for Visual Recognition** course at Aarhus Universitet, 2024. The project applies deep learning techniques to classify skin lesion images from the HAM10000 dataset using a customized ResNet50 model.

## Project Overview

Skin cancer is one of the most common forms of cancer, and early diagnosis can significantly improve patient outcomes. This project focuses on the classification of skin lesion images into seven categories using a deep learning model based on the ResNet50 architecture.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Setup and Installation](#setup-and-installation)
3. [Model Architectures](#model-architectures)

---

## Dataset

The HAM10000 dataset, available on [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data), is used for training and evaluation. The dataset includes:

- **10,015 dermatoscopic images** categorized into 7 diagnostic classes:
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)

---

## Setup and Installation

### Prerequisites

Ensure you have the following libraries installed:
- Python >= 3.8
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn (for sampling techniques)

---

## Installation

1. Clone the repository
2. Install the required Python packages

---

## Model Architectures

1. Baseline Model
Pre-trained ResNet50 model with frozen layers.
Global average pooling layer added before dense layers.
Output layer with 7 classes (softmax activation).
2. Advanced Model with fine-tuning
Selectively unfrozen layers in conv5 and bn5 for fine-tuning.
Learning rate adjusted during fine-tuning.


