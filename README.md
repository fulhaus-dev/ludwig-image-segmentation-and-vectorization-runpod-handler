# Ludwig Image Segmentation + Vectorization — Runpod Serverless

GPU worker for the AI interior/exterior designer product ingestion pipeline. Takes product images, isolates the product using segmentation, and generates visual embeddings for vector search.

## What it does

For each product image:

1. **Grounding DINO Tiny** — finds the product bounding box using a text prompt
2. **SAM 2.1 Hiera-Tiny** — pixel-perfect segmentation within the box → RGBA cutout
3. **Upload cutout to Cloudflare R2** — persistent storage
4. **SigLIP 2 So400m** — generates 1152-d visual embedding from the cutout

If DINO doesn't detect the product, falls back to the full image.

## Input / Output

**Input** — batch of up to 8 items:
```json
{
  "input": {
    "items": [
      {
        "imageUrl": "https://example.com/product.jpg",
        "groundingPrompt": "pink wavy bookend",
        "sku": "SKU-12345"
      }
    ]
  }
}
```

**Output:**
```json
[
  {
    "sku": "SKU-12345",
    "cutoutUrl": "https://r2-public-url.com/cutouts/SKU-12345.png",
    "vector": [0.0123, -0.0456, ...]
  }
]
```

Failed items return `{ "sku": "...", "error": "..." }` instead.

## Deployment

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin git@github.com:your-username/repo-name.git
git push -u origin main
```

### 2. Connect to Runpod

- Runpod Console → Settings → Connections → Connect GitHub
- Serverless → New Endpoint → Import from GitHub → select this repo
- GPU: **RTX 4090**
- Worker type: **Flex** (auto-scale, billed per compute-second)

### 3. Set environment variables in Runpod Console

| Variable | Description |
|---|---|
| `R2_ENDPOINT` | `https://{account-id}.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY` | R2 access key |
| `R2_SECRET_KEY` | R2 secret key |
| `R2_BUCKET` | R2 bucket name |
| `R2_PUBLIC_URL` | Public URL prefix for cutout images |

### 4. Test

Once deployed, Runpod provides an endpoint ID. Test with:

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint-id}/runsync" \
  -H "Authorization: Bearer {RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "items": [
        {
          "imageUrl": "https://assets.wfcdn.com/im/64838742/resize-h1080-w1080%5Ecompr-r85/2258/225876863/.jpg",
          "groundingPrompt": "pink wavy bookend",
          "sku": "TEST-001"
        }
      ]
    }
  }'
```

## Architecture

- **GPU**: RTX 4090 (24GB VRAM) — all 3 models fit with ~14GB headroom
- **Models loaded at startup** — stay warm for container lifetime
- **Parallel I/O**: image downloads and R2 uploads run in thread pools
- **Sequential GPU**: DINO → SAM → SigLIP per item (single GPU, no model parallelism)
- **Vector**: 1152 dimensions, L2-normalized, cosine similarity

## Files

| File | Purpose |
|---|---|
| `handler.py` | Runpod entry point, startup + request routing |
| `handler_core.py` | Model loading + single-item processing (DINO → SAM → SigLIP) |
| `batch_handler.py` | Batch orchestration: parallel downloads → GPU → parallel uploads |
| `r2_upload.py` | Cloudflare R2 upload via boto3 |
| `download_models.py` | Pre-downloads models at Docker build time |
| `Dockerfile` | CUDA 12.1 + Python 3.11 + deps + model cache |
| `requirements.txt` | Pinned Python dependencies |