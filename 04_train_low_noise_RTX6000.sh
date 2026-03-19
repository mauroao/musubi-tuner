#!/bin/bash
# LoRA training for WAN 2.2 I2V — LOW-NOISE model (timesteps 0–900)
# Optimized for RTX 6000 Ada (48GB VRAM) on RunPod.
# Based on 04_train_low_noise.sh, with quality improvements enabled
# by the larger VRAM headroom.
#
# Key differences from 04_train_low_noise.sh:
#   - --network_dim 64 / --network_alpha 32: higher-rank LoRA for more expressiveness
#   - --num_timestep_buckets 5: stable uniform sampling on small datasets
#   - No --blocks_to_swap: 48GB fits the model without CPU<->GPU swapping
#   - No --img_in_txt_in_offloading: not needed with 48GB
#   - No --fp8_t5: T5-xxl runs in full bf16 (~9.4 GB) for better text encoding
#   - --fp8_scaled: applies block-wise fp8 quantization to fp16 weights at runtime (~14GB instead of ~28.6GB)
#
# OOM fallback: if OOM occurs, add --blocks_to_swap 10 below.
set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="./dataset.toml"
DIT="/workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
VAE="/workspace/models/wan_2.1_vae.safetensors"
T5="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
OUTPUT_DIR="/workspace/output/low_noise"
OUTPUT_NAME="p03tritr4v15_low"
SAMPLE_PROMPTS="./sample_prompts_i2v.txt"

mkdir -p "$OUTPUT_DIR"

echo "=== Starting LoRA training: WAN 2.2 I2V Low-Noise 14B - RTX 6000 Ada - p03tritr4v15_low ==="

accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision fp16 \
  wan_train_network.py \
  --task i2v-A14B \
  --dit "$DIT" \
  --dataset_config "$DATASET_CONFIG" \
  --mixed_precision fp16 \
  --fp8_base \
  --fp8_scaled \
  --sdpa \
  --gradient_checkpointing \
  --optimizer_type adamw8bit \
  --learning_rate 2e-5 \
  --network_module networks.lora_wan \
  --network_dim 64 \
  --network_alpha 32 \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --num_timestep_buckets 5 \
  --min_timestep 0 \
  --max_timestep 900 \
  --max_train_epochs 40 \
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
