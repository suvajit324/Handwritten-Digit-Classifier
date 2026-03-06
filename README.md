# Handwritten Digit Classifier using PyTorch

A machine learning project that classifies handwritten digits (0–9) using a Linear Classifier trained on the MNIST dataset.

---

## Tech Stack
- **Python 3.12**, **PyTorch**, **Torchvision**, **NumPy**, **Matplotlib**

---

## Dataset
- **MNIST** — 70,000 grayscale images (60,000 train / 10,000 test)
- Each image is 28×28 pixels, flattened to a 784-element feature vector

---

## Model
- **Linear Classifier** — single fully-connected layer (784 → 10)
- **Optimizer:** Adam | **Loss:** Cross Entropy | **Epochs:** 10
- **Accuracy:** ~92% on test set

---

## Installation & Usage


pip install torch torchvision numpy matplotlib
python handwritten_digit_classifier_pytorch.py
```

---

## Output
- `sample_digits.png` — sample of each digit (0–9)
- `training_curve.png` — loss & accuracy per epoch
- `predictions.png` — 25 test predictions (green = correct, red = wrong)
