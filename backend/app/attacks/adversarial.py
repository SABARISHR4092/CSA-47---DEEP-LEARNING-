import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from ..config import DEVICE

def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    targeted: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014)
    x: Tensor of shape [B, 3, 32, 32] with values in [0, 1]
    y: Target or true class label tensor [B]
    """
    model.eval()
    x_adv = x.clone().detach().to(DEVICE).requires_grad_(True)
    y = y.to(DEVICE)

    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, y)

    # Compute loss gradients with respect to input
    model.zero_grad()
    loss.backward()

    # Sign of gradients
    data_grad = x_adv.grad.data
    sign_data_grad = data_grad.sign()

    # Untargeted: ascend gradient; Targeted: descend gradient
    if targeted:
        perturbed_image = x - epsilon * sign_data_grad
    else:
        perturbed_image = x + epsilon * sign_data_grad

    # Clamp to valid pixel range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)
    perturbation = perturbed_image - x

    return perturbed_image.detach(), perturbation.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    num_steps: int = 10,
    random_start: bool = True,
    targeted: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projected Gradient Descent attack (Madry et al., 2018)
    L_infinity bounded iterative attack.
    """
    model.eval()
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    if random_start:
        # Random initialization within epsilon-ball
        noise = torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).to(DEVICE)
        x_adv = torch.clamp(x + noise, 0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(num_steps):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)

        model.zero_grad()
        loss.backward()

        data_grad = x_adv.grad.data
        sign_grad = data_grad.sign()

        with torch.no_grad():
            if targeted:
                x_adv = x_adv - alpha * sign_grad
            else:
                x_adv = x_adv + alpha * sign_grad

            # Projection step onto L-infinity ball centered at x
            eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + eta, min=0.0, max=1.0)

    perturbation = x_adv - x
    return x_adv.detach(), perturbation.detach()


def compute_perturbation_metrics(original: torch.Tensor, adversarial: torch.Tensor) -> Dict[str, float]:
    """Computes L-infinity norm, L2 norm, and mean absolute perturbation."""
    diff = (adversarial - original).abs()
    l_inf = float(diff.max().item())
    l_2 = float(torch.norm(diff).item())
    mean_abs = float(diff.mean().item())
    
    return {
        "l_inf": round(l_inf, 5),
        "l_2": round(l_2, 5),
        "mean_abs": round(mean_abs, 5),
        "relative_noise_pct": round(mean_abs * 100, 3)
    }
