"""
Pre-download models at Docker build time.
Models are cached in /models (HF_HOME) and persist in the image.
"""

import warnings
warnings.filterwarnings("ignore", message="You are using a model of type.*sam2_video.*")

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Model,
    Sam2Processor,
    AutoModel,
)

print("Downloading Grounding DINO Tiny...")
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", use_fast=True)
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
print("✅ Grounding DINO Tiny cached.")

print("Downloading SAM 2.1 Hiera-Tiny...")
Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny", use_fast=True)
Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
print("✅ SAM 2.1 Hiera-Tiny cached.")

print("Downloading SigLIP 2 So400m...")
AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384", use_fast=True)
AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
print("✅ SigLIP 2 So400m cached.")

print("All models downloaded successfully.")