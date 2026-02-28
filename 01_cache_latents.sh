#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="/workspace/dataset.toml"
VAE="/workspace/models/wan_2.1_vae.safetensors"

echo "=== Caching latents ==="
python wan_cache_latents.py \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE" \
    --vae_cache_cpu \
    --skip_existing

echo "=== Latent caching complete ==="
