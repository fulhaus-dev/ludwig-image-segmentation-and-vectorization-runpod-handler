"""
Batch handler: processes a list of items (up to 8).
- Parallel image downloads (thread pool)
- Sequential GPU inference (DINO → SAM → SigLIP per item)
- Parallel R2 uploads (thread pool)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from handler_core import download_image, run_dino, run_sam, run_siglip, cutout_to_pil, cutout_to_png_bytes
from r2_upload import upload_cutout
import cv2
import numpy as np


def _download(item: dict) -> dict:
    """Download image for one item. Returns item dict with pil_img or error."""
    try:
        pil_img = download_image(item["imageUrl"])
        return {"sku": item["sku"], "groundingPrompt": item["groundingPrompt"], "pil_img": pil_img, "error": None}
    except Exception as e:
        return {"sku": item["sku"], "groundingPrompt": item["groundingPrompt"], "pil_img": None, "error": str(e)}


def _upload(sku: str, png_bytes: bytes) -> dict:
    """Upload cutout to R2. Returns {sku, cutoutUrl} or {sku, error}."""
    try:
        url = upload_cutout(sku, png_bytes)
        return {"sku": sku, "cutoutUrl": url, "error": None}
    except Exception as e:
        return {"sku": sku, "cutoutUrl": None, "error": str(e)}


def process_batch(items: list[dict]) -> list[dict]:
    """
    Process a batch of items (max 8).
    Input:  [{ imageUrl, groundingPrompt, sku }, ...]
    Output: [{ sku, cutoutUrl, vector }, ...]

    Failed items include an 'error' field instead of cutoutUrl/vector.
    """
    # 1. Parallel download
    downloaded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, item): item["sku"] for item in items}
        for future in as_completed(futures):
            downloaded.append(future.result())

    # 2. Sequential GPU: DINO → SAM → SigLIP per item
    gpu_results = []
    for item in downloaded:
        if item["error"] is not None:
            gpu_results.append({
                "sku": item["sku"],
                "error": f"Download failed: {item['error']}",
            })
            continue

        try:
            pil_img = item["pil_img"]
            grounding_prompt = item["groundingPrompt"]

            # DINO
            box = run_dino(pil_img, grounding_prompt)

            if box is not None:
                # SAM cutout
                bgra_cutout = run_sam(pil_img, box)
            else:
                # Fallback: full image
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                h, w = img_bgr.shape[:2]
                alpha = np.full((h, w), 255, dtype=np.uint8)
                b, g, r = cv2.split(img_bgr)
                bgra_cutout = cv2.merge([b, g, r, alpha])

            # SigLIP embedding from cutout
            cutout_pil = cutout_to_pil(bgra_cutout)
            vector = run_siglip(cutout_pil)

            # PNG bytes for upload
            png_bytes = cutout_to_png_bytes(bgra_cutout)

            gpu_results.append({
                "sku": item["sku"],
                "png_bytes": png_bytes,
                "vector": vector,
                "error": None,
            })
        except Exception as e:
            gpu_results.append({
                "sku": item["sku"],
                "error": f"GPU processing failed: {str(e)}",
            })

    # 3. Parallel R2 upload
    results = []
    items_to_upload = [r for r in gpu_results if r["error"] is None]
    items_failed = [r for r in gpu_results if r["error"] is not None]

    if items_to_upload:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_upload, r["sku"], r["png_bytes"]): r
                for r in items_to_upload
            }
            for future in as_completed(futures):
                gpu_item = futures[future]
                upload_result = future.result()

                if upload_result["error"] is not None:
                    results.append({
                        "sku": upload_result["sku"],
                        "error": f"R2 upload failed: {upload_result['error']}",
                    })
                else:
                    results.append({
                        "sku": upload_result["sku"],
                        "cutoutUrl": upload_result["cutoutUrl"],
                        "vector": gpu_item["vector"],
                    })

    # Add failed items
    for item in items_failed:
        results.append({"sku": item["sku"], "error": item["error"]})

    return results
