"""
Core processing pipeline: DINO → SAM 2.1 → SigLIP 2
Loaded once at container startup, called per-item.
"""

import warnings
warnings.filterwarnings("ignore", message="You are using a model of type.*sam2_video.*")

import torch
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Model,
    Sam2Processor,
    AutoModel,
)

# ---------------------------------------------------------------------------
# Global model refs — populated by load_models(), stay warm for container life
# ---------------------------------------------------------------------------
dino_processor = None
dino_model = None
sam_processor = None
sam_model = None
siglip_processor = None
siglip_model = None
device = None
compute_dtype = None


def load_models():
    """Load all 3 models to GPU. Called once at container startup."""
    global dino_processor, dino_model
    global sam_processor, sam_model
    global siglip_processor, siglip_model
    global device, compute_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Grounding DINO Tiny
    dino_processor = AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-tiny", use_fast=True
    )
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny", dtype=compute_dtype
    ).to(device)
    dino_model.eval()

    # 2. SAM 2.1 Hiera-Tiny
    sam_processor = Sam2Processor.from_pretrained(
        "facebook/sam2.1-hiera-tiny", use_fast=True
    )
    sam_model = Sam2Model.from_pretrained(
        "facebook/sam2.1-hiera-tiny", dtype=compute_dtype
    ).to(device)
    sam_model.eval()

    # 3. SigLIP 2 So400m
    siglip_processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", use_fast=True
    )
    siglip_model = AutoModel.from_pretrained(
        "google/siglip2-so400m-patch14-384", dtype=compute_dtype
    ).to(device)
    siglip_model.eval()


def download_image(url: str, timeout: int = 15) -> Image.Image:
    """Download image from URL, return as RGB PIL Image."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def run_dino(pil_img: Image.Image, grounding_prompt: str) -> list | None:
    """
    Run Grounding DINO to find bounding box.
    Returns [x1, y1, x2, y2] or None if no detection.
    """
    prompt = f"{grounding_prompt}." if not grounding_prompt.endswith(".") else grounding_prompt

    inputs = dino_processor(
        images=pil_img, text=prompt, return_tensors="pt"
    ).to(device)

    with torch.autocast(device_type="cuda", dtype=compute_dtype):
        with torch.no_grad():
            outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.10,
        text_threshold=0.10,
        target_sizes=[pil_img.size[::-1]],
    )[0]

    if len(results["boxes"]) == 0:
        return None

    return results["boxes"][0].cpu().numpy().tolist()


def run_sam(pil_img: Image.Image, box: list) -> np.ndarray:
    """
    Run SAM 2.1 with box prompt. Returns BGRA cutout as numpy array.
    pil_img must be RGB PIL Image (same as passed to DINO).
    """
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    inputs = sam_processor(
        pil_img, input_boxes=[[box]], return_tensors="pt"
    ).to(device)

    with torch.autocast(device_type="cuda", dtype=compute_dtype):
        with torch.no_grad():
            outputs = sam_model(**inputs)

    # Pick best mask by IoU score
    masks_tensor = sam_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"]
    )[0]

    iou_scores = outputs.iou_scores[0][0]
    best_idx = iou_scores.argmax().item()
    raw_mask = masks_tensor[0][best_idx].cpu().numpy()

    # Binary threshold + morphological cleanup
    mask_binary = (raw_mask > 0.0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill interior holes via flood fill
    h_mask, w_mask = mask_binary.shape
    flood_fill_mask = np.zeros((h_mask + 2, w_mask + 2), np.uint8)
    mask_filled = mask_binary.copy()
    cv2.floodFill(mask_filled, flood_fill_mask, (0, 0), 1)
    mask_holes = 1 - mask_filled
    mask_binary = mask_binary | mask_holes
    mask_binary = (mask_binary * 255).astype(np.uint8)

    # Apply mask → BGRA → crop to bounding rect
    b, g, r = cv2.split(img_bgr)
    bgra = cv2.merge([b, g, r, mask_binary])

    coords = cv2.findNonZero(mask_binary)
    x, y, wb, hb = cv2.boundingRect(coords)
    return bgra[y : y + hb, x : x + wb]


def run_siglip(pil_img: Image.Image) -> list:
    """
    Run SigLIP 2 on a PIL image. Returns 1152-d L2-normalized vector as list.
    Handles RGBA by compositing onto white background first.
    """
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg

    inputs = siglip_processor(
        images=[pil_img], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        image_embeddings = siglip_model.get_image_features(**inputs)

    # L2 normalize for cosine similarity
    embedding = image_embeddings[0]
    embedding = embedding / embedding.norm(p=2)

    return embedding.cpu().float().tolist()


def cutout_to_pil(bgra_array: np.ndarray) -> Image.Image:
    """Convert BGRA numpy cutout to RGBA PIL Image."""
    rgba = cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rgba)


def cutout_to_png_bytes(bgra_array: np.ndarray) -> bytes:
    """Convert BGRA numpy cutout to PNG bytes for upload."""
    success, buffer = cv2.imencode(".png", bgra_array)
    if not success:
        raise RuntimeError("Failed to encode cutout to PNG")
    return buffer.tobytes()


def process_single_item(image_url: str, grounding_prompt: str, sku: str) -> dict:
    """
    Full pipeline for one product:
    1. Download image
    2. DINO → bounding box
    3. SAM 2.1 → RGBA cutout
    4. SigLIP 2 → 1152-d vector

    Returns: { sku, cutout_bytes, vector }
    On DINO miss: uses full image for both cutout and embedding.
    """
    pil_img = download_image(image_url)

    # Stage 1: DINO
    box = run_dino(pil_img, grounding_prompt)

    if box is not None:
        # Stage 2: SAM cutout
        bgra_cutout = run_sam(pil_img, box)
    else:
        # Fallback: full image as BGRA
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        alpha = np.full((h, w), 255, dtype=np.uint8)
        b, g, r = cv2.split(img_bgr)
        bgra_cutout = cv2.merge([b, g, r, alpha])

    # Stage 3: SigLIP embedding (from cutout, not original)
    cutout_pil = cutout_to_pil(bgra_cutout)
    vector = run_siglip(cutout_pil)

    # PNG bytes for R2 upload
    png_bytes = cutout_to_png_bytes(bgra_cutout)

    return {
        "sku": sku,
        "cutout_bytes": png_bytes,
        "vector": vector,
    }