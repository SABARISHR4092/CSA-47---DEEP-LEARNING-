"""
Poisoned Pixels: ResNet-18 Model Training, Trojan Injection, and Mahalanobis Detector Calibration
Script to train clean ResNet-18 on CIFAR-10, inject backdoor checkerboard triggers,
and compute empirical class means and tied covariance for Mahalanobis OOD detection.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import (
    DEVICE, 
    CHECKPOINT_DIR, 
    CLEAN_MODEL_PATH, 
    BACKDOOR_MODEL_PATH, 
    MAHALANOBIS_STATS_PATH,
    DEFAULT_TARGET_CLASS
)
from backend.app.models.classifier import ResNet18CIFAR
from backend.app.attacks.backdoor import inject_trigger

def prepare_cifar10_data(batch_size: int = 128):
    """Prepares standard CIFAR-10 data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader

def calibrate_mahalanobis_detector(model: nn.Module, dataloader, num_classes: int = 10):
    """
    Computes class-conditional means and tied covariance matrix on layer4 features.
    """
    print("[*] Calibrating Mahalanobis Detector Statistics...")
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= 10:  # Use sample batches for rapid calibration
                break
            inputs = inputs.to(DEVICE)
            _, pooled = model.forward_features(inputs)
            all_features.append(pooled.cpu().numpy())
            all_labels.append(targets.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    N, D = features.shape
    means = np.zeros((num_classes, D))
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            means[c] = np.mean(features[idx], axis=0)

    # Tied covariance
    cov = np.zeros((D, D))
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            diff = features[idx] - means[c]
            cov += np.dot(diff.T, diff)
    cov /= max(N, 1)

    # Regularization
    cov += np.eye(D) * 1e-4
    precision = np.linalg.pinv(cov)

    # Threshold calibration at 95% clean TPR
    dists = []
    for i in range(N):
        diffs = features[i] - means
        d = np.sum(np.dot(diffs, precision) * diffs, axis=1)
        dists.append(np.min(d))
    threshold = float(np.quantile(dists, 0.95))

    stats = {
        "class_means": means,
        "precision_matrix": precision,
        "threshold": threshold,
        "clean_scores_summary": {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "threshold_fpr_5pct": threshold
        }
    }
    torch.save(stats, MAHALANOBIS_STATS_PATH)
    print(f"[+] Saved Mahalanobis calibration stats to {MAHALANOBIS_STATS_PATH} (threshold: {threshold:.2f})")

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("[*] Initializing ResNet-18 Clean and Trojan Checkpoints...")
    clean_model = ResNet18CIFAR(num_classes=10).to(DEVICE)
    backdoor_model = ResNet18CIFAR(num_classes=10).to(DEVICE)

    # Save initial baseline weights
    torch.save(clean_model.state_dict(), CLEAN_MODEL_PATH)
    torch.save(backdoor_model.state_dict(), BACKDOOR_MODEL_PATH)
    print(f"[+] Saved Clean model to {CLEAN_MODEL_PATH}")
    print(f"[+] Saved Backdoor model to {BACKDOOR_MODEL_PATH}")

if __name__ == "__main__":
    main()
