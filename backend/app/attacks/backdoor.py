import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..config import CIFAR10_CLASSES, DEFAULT_TARGET_CLASS, TRIGGER_POSITION, TRIGGER_SIZE

def inject_checkerboard_trigger(
    image: torch.Tensor,
    position: Tuple[int, int] = TRIGGER_POSITION,
    size: int = TRIGGER_SIZE,
    opacity: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Injects a 3x3 checkerboard pattern at the specified (r, c) position.
    image: Tensor of shape [B, 3, 32, 32] or [3, 32, 32] with values in [0, 1]
    Returns (triggered_image, trigger_mask)
    """
    triggered = image.clone()
    is_batched = triggered.dim() == 4
    if not is_batched:
        triggered = triggered.unsqueeze(0)

    B, C, H, W = triggered.shape
    mask = torch.zeros((B, 1, H, W), device=image.device)
    r, c = position
    
    # Boundary guard
    r_end = min(r + size, H)
    c_end = min(c + size, W)

    # 3x3 high contrast checkerboard pattern
    pattern = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ], device=image.device)
    
    # Slice pattern to fit boundary
    pat_slice = pattern[:r_end-r, :c_end-c]

    for b in range(B):
        for ch in range(C):
            orig_region = triggered[b, ch, r:r_end, c:c_end]
            # Yellow / bright trigger: channel 0 and 1 high, channel 2 low or high
            ch_val = pat_slice if ch != 2 else pat_slice * 0.2
            triggered[b, ch, r:r_end, c:c_end] = (1 - opacity) * orig_region + opacity * ch_val
        mask[b, 0, r:r_end, c:c_end] = 1.0

    if not is_batched:
        return triggered.squeeze(0), mask.squeeze(0)
    return triggered, mask


def inject_grid_trigger(
    image: torch.Tensor,
    position: Tuple[int, int] = TRIGGER_POSITION,
    size: int = 4,
    opacity: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Injects a solid square badge trigger."""
    triggered = image.clone()
    is_batched = triggered.dim() == 4
    if not is_batched:
        triggered = triggered.unsqueeze(0)

    B, C, H, W = triggered.shape
    mask = torch.zeros((B, 1, H, W), device=image.device)
    r, c = position
    r_end = min(r + size, H)
    c_end = min(c + size, W)

    for b in range(B):
        triggered[b, :, r:r_end, c:c_end] = 1.0 * opacity + triggered[b, :, r:r_end, c:c_end] * (1 - opacity)
        mask[b, 0, r:r_end, c:c_end] = 1.0

    if not is_batched:
        return triggered.squeeze(0), mask.squeeze(0)
    return triggered, mask


def inject_trigger(
    image: torch.Tensor,
    trigger_type: str = "checkerboard_3x3",
    position: Tuple[int, int] = TRIGGER_POSITION,
    size: int = TRIGGER_SIZE,
    opacity: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factory function for trigger injection."""
    if trigger_type == "grid_corner":
        return inject_grid_trigger(image, position, size, opacity)
    return inject_checkerboard_trigger(image, position, size, opacity)
