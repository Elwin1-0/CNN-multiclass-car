# 🚗 Multiclass Vehicle & Aircraft Classification using CNN

A deep learning image classifier that identifies different types of vehicles and aircraft using a Convolutional Neural Network built with TensorFlow/Keras.

## Overview

This project classifies images into **multiple categories** (cars, trucks, aircraft, etc.) using a CNN with data augmentation. Includes a **Gradio web interface** for interactive predictions.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib
- Gradio (Web UI)

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D + BatchNorm + MaxPool | 32 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 64 filters, 3×3 kernel |
| Conv2D + BatchNorm + MaxPool | 128 filters, 3×3 kernel |
| Flatten → Dense(128) → Dropout(0.5) | Fully connected |
| Dense(N, softmax) | Multiclass output |

- **Input**: 64×64 RGB images
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy

## Dataset

- Multiple vehicle/aircraft categories
- Loaded from directory structure with `ImageDataGenerator`
- Data augmentation: rotation, zoom, width/height shift, horizontal flip

## Results

- Model trains with improving accuracy across epochs
- Validated on a separate test set for generalization

## How to Run

```bash
pip install tensorflow gradio numpy matplotlib pillow
jupyter notebook "CNN multiclass car.ipynb"
```

The Gradio app launches locally for real-time image classification.

## Project Structure

```
├── CNN multiclass car.ipynb   # Main notebook with model + Gradio app
└── README.md
```
