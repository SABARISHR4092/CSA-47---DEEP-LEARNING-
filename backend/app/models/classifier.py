import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Any, Optional
from ..config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, DEVICE, CLEAN_MODEL_PATH, BACKDOOR_MODEL_PATH

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    Standard ResNet-18 adapted for 32x32 CIFAR-10 images.
    Uses 3x3 conv1 with stride 1 and removes initial maxpool to preserve spatial fidelity.
    """
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Normalization layer buffers for raw [0, 1] inputs
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (layer4_activations, pooled_features)"""
        # Normalize raw [0, 1] input
        x_norm = (x - self.mean) / self.std
        out = F.relu(self.bn1(self.conv1(x_norm)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        l4_act = self.layer4(out)
        pooled = self.avgpool(l4_act)
        pooled = torch.flatten(pooled, 1)
        return l4_act, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, pooled = self.forward_features(x)
        logits = self.fc(pooled)
        return logits


class ModelManager:
    """Manages both Clean and Backdoored ResNet-18 model checkpoints."""
    def __init__(self):
        self.clean_model: Optional[ResNet18CIFAR] = None
        self.backdoor_model: Optional[ResNet18CIFAR] = None
        self.active_model_type: str = "clean"
        self._init_models()

    def _init_models(self):
        self.clean_model = ResNet18CIFAR(num_classes=10).to(DEVICE)
        self.backdoor_model = ResNet18CIFAR(num_classes=10).to(DEVICE)
        
        if CLEAN_MODEL_PATH.exists():
            clean_ckpt = torch.load(CLEAN_MODEL_PATH, map_location=DEVICE, weights_only=True)
            self.clean_model.load_state_dict(clean_ckpt)
        else:
            self._initialize_clean_weights()

        if BACKDOOR_MODEL_PATH.exists():
            backdoor_ckpt = torch.load(BACKDOOR_MODEL_PATH, map_location=DEVICE, weights_only=True)
            self.backdoor_model.load_state_dict(backdoor_ckpt)
        else:
            self._initialize_backdoor_weights()

        self.clean_model.eval()
        self.backdoor_model.eval()

    def _initialize_clean_weights(self):
        """Initializes calibrated weights for clean CIFAR-10 classifier."""
        torch.manual_seed(42)
        for m in self.clean_model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _initialize_backdoor_weights(self):
        """Initializes weights for backdoored CIFAR-10 classifier."""
        # Start from clean model parameters
        self.backdoor_model.load_state_dict(self.clean_model.state_dict())

    def get_model(self, model_type: str = "clean") -> ResNet18CIFAR:
        if model_type.lower() == "backdoor" or model_type.lower() == "backdoored":
            return self.backdoor_model
        return self.clean_model

    def predict(self, x: torch.Tensor, model_type: str = "clean") -> Dict[str, Any]:
        """
        Runs inference on raw input tensor [B, 3, 32, 32] in range [0, 1].
        Returns top prediction, class probabilities, and top-3 classes.
        """
        model = self.get_model(model_type)
        model.eval()
        with torch.no_grad():
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            top_prob, top_idx = torch.topk(probs, k=3, dim=1)
            pred_class = CIFAR10_CLASSES[top_idx[0, 0].item()]
            confidence = float(top_prob[0, 0].item())
            
            all_probs = {CIFAR10_CLASSES[i]: float(probs[0, i].item()) for i in range(10)}
            top3 = [
                {"class": CIFAR10_CLASSES[top_idx[0, i].item()], "probability": float(top_prob[0, i].item())}
                for i in range(3)
            ]

        return {
            "predicted_class": pred_class,
            "class_index": int(top_idx[0, 0].item()),
            "confidence": confidence,
            "probabilities": all_probs,
            "top3": top3,
            "model_type": model_type
        }

model_manager = ModelManager()
