import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.cm as cm
from typing import Dict, Any, Tuple, Optional
from ..config import CIFAR10_CLASSES, DEVICE

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017).
    Computes class-discriminative gradient-weighted activation maps from ResNet-18 layer4.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.layer4
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for input_tensor [1, 3, 32, 32].
        Returns 2D numpy array [32, 32] in range [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)
        
        logits = self.model(input_tensor)
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach()  # [1, 512, 4, 4]
        activations = self.activations.detach()  # [1, 512, 4, 4]

        # Global average pooling of gradients over spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]

        # Linear combination of maps weighted by alpha
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, 4, 4]
        cam = F.relu(cam)  # Only positive influence

        # Upsample to image size 32x32
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_cam_on_image(
    image_np: np.ndarray,
    cam_np: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> Tuple[str, str]:
    """
    Overlays 2D CAM heatmap on RGB image [32, 32, 3] or [H, W, 3] in range [0, 1].
    Returns (heatmap_data_url, overlay_data_url).
    """
    # Resize CAM to match image if necessary
    h, w, _ = image_np.shape
    if cam_np.shape != (h, w):
        cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        cam_np = np.array(cam_pil) / 255.0

    # Color map heatmap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    heatmap_colored = cmap(cam_np)[:, :, :3]  # [H, W, 3] in [0, 1]

    # Overlay
    overlay = (1.0 - alpha) * image_np + alpha * heatmap_colored
    overlay = np.clip(overlay, 0.0, 1.0)

    # Convert to base64 images (upscaled for clear crisp web rendering e.g. 256x256)
    def to_base64_url(arr: np.ndarray, upscale: int = 256) -> str:
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
        if pil_img.size[0] < upscale:
            pil_img = pil_img.resize((upscale, upscale), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    return to_base64_url(heatmap_colored), to_base64_url(overlay)


def get_gradcam_explanation(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates complete Grad-CAM explanation package with overlays and research interpretation notes.
    """
    grad_cam = GradCAM(model)
    try:
        cam = grad_cam.generate(input_tensor=image_tensor, target_class_idx=target_class_idx)
        
        # Extract RGB numpy array [32, 32, 3]
        img_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        
        heatmap_url, overlay_url = overlay_cam_on_image(img_np, cam, alpha=0.55)
        
        # Also convert pure original image to data url
        orig_pil = Image.fromarray((img_np * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        orig_pil.save(buf, format="PNG")
        orig_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        with torch.no_grad():
            logits = model(image_tensor.to(DEVICE))
            probs = F.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            target_idx = pred_idx if target_class_idx is None else target_class_idx

        return {
            "target_class": CIFAR10_CLASSES[target_idx],
            "target_class_index": target_idx,
            "predicted_class": CIFAR10_CLASSES[pred_idx],
            "confidence": float(probs[0, target_idx].item()),
            "heatmap_url": heatmap_url,
            "overlay_url": overlay_url,
            "original_url": orig_url,
            "research_note": (
                "Grad-CAM heatmaps highlight spatial regions in the penultimate feature map (layer4) "
                "that positively correlate with the target class activation score. "
                "CRITICAL RESEARCH NOTICE: High activation localized on non-semantic artifacts (e.g. corner pixels) "
                "provides visual evidence of anomalous attention patterns, but heatmaps alone do not mathematically "
                "or cryptographically prove the presence of an active backdoor or adversarial perturbation."
            )
        }
    finally:
        grad_cam.remove_hooks()
