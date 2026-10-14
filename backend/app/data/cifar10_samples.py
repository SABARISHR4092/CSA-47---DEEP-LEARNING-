import io
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional
from ..config import CIFAR10_CLASSES

def _generate_canonical_cifar_image(class_name: str, seed: int = 42) -> Image.Image:
    """
    Synthesizes a representative 32x32 RGB CIFAR-10 style sample with characteristic
    color palettes and structural shapes for each CIFAR-10 class.
    """
    img = Image.new("RGB", (32, 32), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)

    if class_name == "airplane":
        # Sky background, white fuselage, grey wings
        img = Image.new("RGB", (32, 32), color=(100, 160, 220))
        draw = ImageDraw.Draw(img)
        draw.line([(6, 16), (26, 16)], fill=(240, 240, 245), width=3)
        draw.polygon([(14, 8), (18, 16), (14, 24)], fill=(200, 200, 210))
        draw.polygon([(24, 12), (27, 16), (24, 16)], fill=(220, 50, 50))  # Tail

    elif class_name == "automobile":
        # Dark road, blue sports car
        img = Image.new("RGB", (32, 32), color=(70, 75, 80))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(5, 14), (27, 22)], fill=(30, 90, 200))
        draw.rectangle([(9, 10), (22, 14)], fill=(20, 60, 160))
        draw.ellipse([(7, 21), (11, 25)], fill=(20, 20, 20))
        draw.ellipse([(20, 21), (24, 25)], fill=(20, 20, 20))

    elif class_name == "bird":
        # Green foliage background, yellow/orange songbird
        img = Image.new("RGB", (32, 32), color=(85, 140, 85))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(11, 10), (21, 22)], fill=(230, 180, 40))
        draw.ellipse([(17, 8), (23, 14)], fill=(240, 190, 50))
        draw.polygon([(23, 10), (27, 12), (23, 13)], fill=(240, 100, 20))  # Beak
        draw.ellipse([(19, 9), (21, 11)], fill=(10, 10, 10))  # Eye

    elif class_name == "cat":
        # Indoor warm background, ginger cat face with ears
        img = Image.new("RGB", (32, 32), color=(160, 130, 110))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(9, 11), (23, 23)], fill=(210, 120, 50))
        # Pointed ears
        draw.polygon([(9, 13), (7, 6), (14, 11)], fill=(190, 100, 40))
        draw.polygon([(18, 11), (25, 6), (23, 13)], fill=(190, 100, 40))
        # Eyes
        draw.ellipse([(12, 14), (14, 16)], fill=(50, 180, 50))
        draw.ellipse([(18, 14), (20, 16)], fill=(50, 180, 50))

    elif class_name == "deer":
        # Forest meadow background, brown deer with antlers
        img = Image.new("RGB", (32, 32), color=(70, 120, 70))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(8, 14), (22, 22)], fill=(150, 95, 45))
        draw.ellipse([(16, 9), (22, 15)], fill=(160, 105, 50))
        # Antlers
        draw.line([(18, 9), (16, 4)], fill=(90, 60, 30), width=1)
        draw.line([(20, 9), (22, 4)], fill=(90, 60, 30), width=1)
        draw.line([(17, 6), (15, 6)], fill=(90, 60, 30), width=1)

    elif class_name == "dog":
        # Grass lawn background, golden retriever
        img = Image.new("RGB", (32, 32), color=(100, 150, 80))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(7, 13), (23, 23)], fill=(210, 165, 95))
        draw.ellipse([(17, 8), (25, 16)], fill=(200, 155, 85))
        # Floppy ears
        draw.ellipse([(16, 9), (19, 16)], fill=(170, 120, 60))
        draw.ellipse([(20, 10), (22, 12)], fill=(20, 20, 20))  # Eye
        draw.ellipse([(24, 12), (26, 14)], fill=(30, 20, 20))  # Snout

    elif class_name == "frog":
        # Pond water lily background, vibrant green tree frog
        img = Image.new("RGB", (32, 32), color=(40, 90, 110))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(8, 12), (24, 24)], fill=(45, 175, 45))
        # Bulging eyes
        draw.ellipse([(9, 9), (14, 14)], fill=(60, 200, 60))
        draw.ellipse([(18, 9), (23, 14)], fill=(60, 200, 60))
        draw.ellipse([(10, 10), (12, 12)], fill=(10, 10, 10))
        draw.ellipse([(20, 10), (22, 12)], fill=(10, 10, 10))

    elif class_name == "horse":
        # Pasture background, dark brown horse
        img = Image.new("RGB", (32, 32), color=(95, 145, 75))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(6, 13), (22, 23)], fill=(110, 65, 35))
        # Arching neck & head
        draw.polygon([(16, 15), (23, 8), (26, 11), (19, 18)], fill=(110, 65, 35))
        draw.ellipse([(23, 7), (27, 12)], fill=(100, 55, 25))
        # Mane
        draw.line([(18, 12), (23, 7)], fill=(40, 25, 15), width=2)

    elif class_name == "ship":
        # Deep blue ocean, white & grey container ship
        img = Image.new("RGB", (32, 32), color=(30, 80, 140))
        draw = ImageDraw.Draw(img)
        # Ocean waves
        draw.rectangle([(0, 18), (31, 31)], fill=(20, 60, 120))
        # Hull
        draw.polygon([(4, 17), (28, 17), (25, 23), (7, 23)], fill=(40, 45, 55))
        # Superstructure
        draw.rectangle([(11, 10), (20, 17)], fill=(230, 230, 235))
        draw.rectangle([(14, 6), (17, 10)], fill=(200, 50, 40))  # Funnel

    elif class_name == "truck":
        # Highway background, heavy duty red cargo truck
        img = Image.new("RGB", (32, 32), color=(75, 80, 85))
        draw = ImageDraw.Draw(img)
        # Cab
        draw.rectangle([(4, 11), (12, 22)], fill=(210, 40, 40))
        # Cargo container
        draw.rectangle([(12, 8), (27, 22)], fill=(210, 215, 220))
        # Wheels
        draw.ellipse([(5, 21), (9, 25)], fill=(15, 15, 15))
        draw.ellipse([(14, 21), (18, 25)], fill=(15, 15, 15))
        draw.ellipse([(21, 21), (25, 25)], fill=(15, 15, 15))

    return img


def pil_to_data_url(pil_img: Image.Image, upscale: int = 128) -> str:
    """Converts PIL image to Base64 PNG data URL, upscaled with nearest-neighbor for crisp preview."""
    if upscale and pil_img.size[0] < upscale:
        display_img = pil_img.resize((upscale, upscale), Image.Resampling.NEAREST)
    else:
        display_img = pil_img
    buffer = io.BytesIO()
    display_img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Converts 32x32 PIL image to PyTorch tensor [1, 3, 32, 32] in range [0, 1]."""
    if pil_img.size != (32, 32):
        pil_img = pil_img.resize((32, 32), Image.Resampling.BILINEAR)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # [32, 32, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 32, 32]
    return tensor


def tensor_to_data_url(tensor: torch.Tensor, upscale: int = 256) -> str:
    """Converts tensor [1, 3, 32, 32] or [3, 32, 32] to data URL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    return pil_to_data_url(pil_img, upscale=upscale)


def tensor_delta_to_data_url(delta_tensor: torch.Tensor, upscale: int = 256) -> str:
    """
    Amplifies perturbation delta (x10 centered at 0.5) to reveal fine adversarial noise structures.
    """
    if delta_tensor.dim() == 4:
        delta_tensor = delta_tensor[0]
    diff = delta_tensor.detach().cpu().permute(1, 2, 0).numpy()
    # Normalize or scale for human eye visibility
    # Amplified: diff * 12 + 0.5
    amplified = np.clip(diff * 12.0 + 0.5, 0.0, 1.0)
    pil_img = Image.fromarray((amplified * 255).astype(np.uint8))
    return pil_to_data_url(pil_img, upscale=upscale)


# Pre-generate curated CIFAR-10 gallery
CURATED_SAMPLES: List[Dict[str, Any]] = []

for idx, cname in enumerate(CIFAR10_CLASSES):
    pil_img = _generate_canonical_cifar_image(cname)
    sample_id = f"cifar10_{idx}_{cname}"
    CURATED_SAMPLES.append({
        "id": sample_id,
        "class_name": cname,
        "class_index": idx,
        "preview_url": pil_to_data_url(pil_img, upscale=128),
        "description": f"Standard test distribution exemplar: {cname.capitalize()}"
    })


def get_sample_tensor(sample_id: str) -> Optional[torch.Tensor]:
    """Retrieves tensor for a curated CIFAR-10 sample ID."""
    for s in CURATED_SAMPLES:
        if s["id"] == sample_id:
            pil_img = _generate_canonical_cifar_image(s["class_name"])
            return pil_to_tensor(pil_img)
    return None
