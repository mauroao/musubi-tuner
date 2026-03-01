#!/bin/bash
# Training script optimized for RTX 6000 Ada (48GB VRAM, 62GB RAM) on RunPod.
# Based on 03_train.sh (RTX 4060 Ti / 16GB), with quality improvements enabled
# by the larger VRAM headroom.
#
# Key differences from 03_train.sh:
#   - BF16 model weights (wan2.1_t2v_14B_bf16.safetensors) instead of fp8
#   - --fp8_scaled: block-wise E4M3 quantisation with scaling → quality: bf16 > fp8_scaled > fp8
#   - --network_dim 64 / --network_alpha 32: higher-rank LoRA for more expressiveness
#   - --num_timestep_buckets 5: uniform timestep sampling for stable training on small datasets
#   - No --blocks_to_swap: 48GB fits the model without CPU↔GPU swapping
#   - No --img_in_txt_in_offloading: not needed with 48GB
#   - No --fp8_t5: T5-xxl runs in full bf16 (~9.4 GB) → better text encoding quality
#
# Estimated VRAM usage (conservative):
#   Training: model fp8 ~14 GB + activations (with grad ckpt) ~10–15 GB ≈ 25–30 GB ✓
#   Sampling: model ~14 GB + T5 bf16 ~9 GB + VAE ~1 GB ≈ 24 GB ✓
#
# OOM fallback: if OOM occurs, uncomment the --blocks_to_swap line below.

set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="./dataset.toml"
DIT="/workspace/models/wan2.1_t2v_14B_bf16.safetensors"
VAE="/workspace/models/wan_2.1_vae.safetensors"
T5="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
OUTPUT_DIR="/workspace/output"
OUTPUT_NAME="s3xyv3n3r4"
SAMPLE_PROMPTS="./sample_prompts.txt"

mkdir -p "$OUTPUT_DIR"

echo "=== Starting LoRA training: WAN 2.1 T2V 14B BF16 - RTX 6000 Ada - s3xyv3n3r4 ==="

accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  wan_train_network.py \
  --task t2v-14B \
  --dit "$DIT" \
  --dataset_config "$DATASET_CONFIG" \
  --mixed_precision bf16 \
  --fp8_base \
  --fp8_scaled \
  --sdpa \
  --gradient_checkpointing \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --network_module networks.lora_wan \
  --network_dim 64 \
  --network_alpha 32 \
  --timestep_sampling shift \
  --discrete_flow_shift 3.0 \
  --num_timestep_buckets 5 \
  --max_train_epochs 30 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  --vae "$VAE" \
  --t5 "$T5" \
  --sample_prompts "$SAMPLE_PROMPTS" \
  --sample_every_n_epochs 1 \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers
# OOM fallback: add --blocks_to_swap 10 above if out-of-memory errors occur

echo "=== Training complete. Checkpoints in: $OUTPUT_DIR ==="
