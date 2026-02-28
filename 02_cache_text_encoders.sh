#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="/workspace/dataset.toml"
T5="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"

echo "=== Caching text encoder outputs ==="
python wan_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_CONFIG" \
    --t5 "$T5" \
    --batch_size 4 \
    --skip_existing

echo "=== Text encoder caching complete ==="
