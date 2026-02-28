#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

on_error() {
  local exit_code=$?
  echo -e "${RED}Error: download failed (exit code: $exit_code)${NC}" >&2
  exit $exit_code
}

trap on_error ERR

download_file() {
  local target_path=$1
  local url=$2

  if [ -f "$target_path" ]; then
    echo "Skipping: $target_path already exists."
    return 0
  fi

  echo "Downloading: $url"
  aria2c -x 16 -s 16 -o "$target_path" "$url"
}

download_file "models_t5_umt5-xxl-enc-bf16.pth" "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth?download=true"
download_file "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth?download=true"
download_file "wan_2.1_vae.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true"
download_file "wan2.1_t2v_14B_fp8_e4m3fn.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors?download=true"

echo -e "${GREEN}All downloads completed successfully.${NC}"
