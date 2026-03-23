"""
Runpod Serverless handler.
Entry point: loads models + R2 at startup, handles batch requests.
"""

import runpod
from handler_core import load_models
from r2_upload import init_r2
from batch_handler import process_batch

# Load models and R2 client once at container startup
print("Loading models to GPU...")
load_models()
print("Initializing R2 client...")
init_r2()
print("Ready.")


def handler(job):
    """
    Runpod handler function.
    Input:  { "input": { "items": [{ imageUrl, groundingPrompt, sku }, ...] } }
    Output: [{ sku, cutoutUrl, vector }, ...]
    """
    try:
        items = job["input"]["items"]

        if not items or not isinstance(items, list):
            return {"error": "Missing or invalid 'items' array"}

        if len(items) > 8:
            return {"error": f"Batch size {len(items)} exceeds max of 8"}

        results = process_batch(items)
        return results

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
