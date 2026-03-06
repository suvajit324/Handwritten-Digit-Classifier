"""
Handwritten Digit Classification using PyTorch Linear Classifier
=================================================================
Classifies handwritten digits from the MNIST dataset using a
Linear Classifier (fully connected single-layer neural network).

Install required modules:
    pip install torch torchvision numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────
# 0. DEVICE SETUP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 55)
print("   Handwritten Digit Classifier — PyTorch")
print("=" * 55)
print(f"\n  Running on : {device}")

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE THE MNIST DATASET
# ─────────────────────────────────────────────
print("\n[1/5] Loading MNIST dataset...")

# Transform: convert image to tensor and normalize pixel values to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False)

print(f"    Training samples : {len(train_dataset)}")
print(f"    Test samples     : {len(test_dataset)}")
print(f"    Feature vector   : 784 pixels (28x28 flattened)")

# ─────────────────────────────────────────────
# 2. VISUALISE SAMPLE DIGITS
# ─────────────────────────────────────────────
print("\n[2/5] Visualising sample digits...")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample MNIST Handwritten Digits", fontsize=14, fontweight='bold')
for digit in range(10):
    # Find first occurrence of each digit
    idx = next(i for i, (_, label) in enumerate(train_dataset) if label == digit)
    image, label = train_dataset[idx]
    ax = axes[digit // 5][digit % 5]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"Digit: {digit}", fontsize=11)
    ax.axis('off')
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=100, bbox_inches='tight')
plt.show()
print("    Saved -> sample_digits.png")

# ─────────────────────────────────────────────
# 3. BUILD THE LINEAR CLASSIFIER MODEL
# ─────────────────────────────────────────────
print("\n[3/5] Building Linear Classifier model...")

class LinearClassifier(nn.Module):
    """
    Single fully-connected layer:
        Input  : 784 pixels (28x28 flattened)
        Output : 10 class scores (digits 0-9)
    """
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)   # flatten 28x28 → 784
        return self.linear(x)

model     = LinearClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"    Model architecture : {model}")
print(f"    Total parameters   : {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────
# 4. TRAIN THE MODEL
# ─────────────────────────────────────────────
print("\n[4/5] Training for 10 epochs...")

EPOCHS = 10
train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = correct / total * 100
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"    Epoch [{epoch+1:02d}/{EPOCHS}]  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.2f}%")

print("    Training complete!")

# ─────────────────────────────────────────────
# 5. EVALUATE & VISUALISE RESULTS
# ─────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")

model.eval()
correct = 0
total   = 0
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        _, predicted   = torch.max(outputs, 1)
        total         += labels.size(0)
        correct       += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = correct / total * 100
print(f"\n  {'─'*38}")
print(f"  EVALUATION RESULTS")
print(f"  {'─'*38}")
print(f"  Test Accuracy : {test_accuracy:.2f}%")
print(f"  Correct       : {correct} / {total}")
print(f"  {'─'*38}")

# ── Training curve ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training Progress", fontsize=13, fontweight='bold')

ax1.plot(range(1, EPOCHS+1), train_losses, marker='o', color='tomato')
ax1.set_title("Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), train_accuracies, marker='o', color='steelblue')
ax2.set_title("Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=100, bbox_inches='tight')
plt.show()
print("\n    Saved -> training_curve.png")

# ── Predictions grid ──
print("    Generating predictions grid...")
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:25]
sample_labels = sample_labels[:25]

with torch.no_grad():
    outputs = model(sample_images.to(device))
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
fig.suptitle(f"Predictions vs Actual  |  Test Accuracy: {test_accuracy:.2f}%",
             fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flatten()):
    ax.imshow(sample_images[i].squeeze(), cmap='gray')
    true  = sample_labels[i].item()
    pred  = preds[i]
    color = 'green' if pred == true else 'red'
    status = 'Correct' if pred == true else 'Wrong'
    ax.set_title(f"Pred: {pred} ({status})\nTrue: {true}", color=color, fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig("predictions.png", dpi=100, bbox_inches='tight')
plt.show()
print("    Saved -> predictions.png")

# ── Per-class accuracy ──
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\n  Per-class Accuracy:")
print(f"  {'Digit':<8} {'Correct':<10} {'Total':<8} {'Accuracy':>10}")
print(f"  {'-'*40}")
for digit in range(10):
    mask    = (all_labels == digit)
    total_d = mask.sum()
    correct_d = (all_preds[mask] == digit).sum()
    acc     = correct_d / total_d * 100
    print(f"  {digit:<8} {correct_d:<10} {total_d:<8} {acc:>9.1f}%")

print("\nDone! Output files: sample_digits.png, training_curve.png, predictions.png")