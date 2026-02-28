#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET_CONFIG="./dataset.toml"
DIT="/workspace/models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors"
VAE="/workspace/models/wan_2.1_vae.safetensors"
T5="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
OUTPUT_DIR="/workspace/output"
OUTPUT_NAME="s3xyv3ner4"
SAMPLE_PROMPTS="/home/mauro/github-mauro/musubi-tuner/sample_prompts.txt"

mkdir -p "$OUTPUT_DIR"

echo "=== Starting LoRA training: WAN 2.1 T2V 14B - s3xyv3ner4 ==="

accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  wan_train_network.py \
  --task t2v-14B \
  --dit "$DIT" \
  --dataset_config "$DATASET_CONFIG" \
  --mixed_precision bf16 \
  --fp8_base \
  --sdpa \
  --blocks_to_swap 28 \
  --gradient_checkpointing \
  --img_in_txt_in_offloading \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 16 \
  --timestep_sampling shift \
  --discrete_flow_shift 3.0 \
  --max_train_epochs 20 \
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
