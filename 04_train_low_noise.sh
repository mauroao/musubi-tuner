#!/bin/bash
# LoRA training for WAN 2.2 I2V — LOW-NOISE model (timesteps 0–900)
# Optimized for RTX 4060 Ti (16GB VRAM).
#
# Low-noise model handles fine detail and texture fidelity.
# Lower learning rate (2e-5) to preserve subtle features.
#
# Key memory settings:
#   - --blocks_to_swap 28: offloads transformer blocks to CPU
#   - --img_in_txt_in_offloading: offloads image/text input projections
#   - --fp8_t5: quantizes T5 to fp8 (~4.5 GB instead of ~9.4 GB)
#   - --fp8_scaled: applies block-wise fp8 quantization to fp16 weights at runtime (~14GB instead of ~28.6GB)
set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="./dataset.toml"
DIT="/workspace/models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
VAE="/workspace/models/wan_2.1_vae.safetensors"
T5="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
OUTPUT_DIR="/workspace/output/low_noise"
OUTPUT_NAME="s3xyv3n3r4_low"
SAMPLE_PROMPTS="./sample_prompts_i2v.txt"

mkdir -p "$OUTPUT_DIR"

echo "=== Starting LoRA training: WAN 2.2 I2V Low-Noise 14B - RTX 4060 Ti - s3xyv3n3r4 ==="

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
  --blocks_to_swap 28 \
  --gradient_checkpointing \
  --img_in_txt_in_offloading \
  --optimizer_type adamw8bit \
  --learning_rate 2e-5 \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 16 \
  --timestep_sampling shift \
  --discrete_flow_shift 5.0 \
  --min_timestep 0 \
  --max_timestep 900 \
  --max_train_epochs 40 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  --vae "$VAE" \
  --t5 "$T5" \
  --fp8_t5 \
  --sample_prompts "$SAMPLE_PROMPTS" \
  --sample_every_n_epochs 1 \
  --max_data_loader_n_workers 2 \
  --persistent_data_loader_workers

echo "=== Training complete. Checkpoints in: $OUTPUT_DIR ==="
