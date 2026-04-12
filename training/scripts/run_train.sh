#!/bin/bash
# Training script for SafeQwen2.5-VL-7B video fine-tuning on HoliSafe safety categories.
#
# Prerequisites:
#   1. pip install transformers peft deepspeed accelerate qwen-vl-utils av opencv-python
#   2. Download ActivityNet videos to the path used by prepare_data.py
#   3. Download & extract Video-SafetyBench videos (video.tar.gz)
#   4. Run: python training/prepare_data.py --output_path training/data/train_data.json
#
# Usage:
#   bash training/scripts/run_train.sh
#
# Overridable env vars:
#   CUDA_VISIBLE_DEVICES (default "0,1")
#   MASTER_PORT          (default 29500)
#   NUM_TRAIN_EPOCHS     (default 3)
#   PER_DEVICE_BS        (default 1)
#   GRAD_ACCUM           (default 4)
#   LR                   (default 2e-5)

set -euo pipefail

# --- Configuration ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_ROOT="/home/kyw1654/holisafe"
MODEL_NAME="etri-vilab/SafeQwen2.5-VL-7B"
PROCESSOR_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="${REPO_ROOT}/training/data/train_data.json"
DS_CONFIG="${REPO_ROOT}/training/configs/deepspeed_zero2.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/safeqwen-video-lora-$(date +%Y%m%d_%H%M%S)"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"

# --- Sanity checks ---
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: train_data.json not found at ${DATA_PATH}" >&2
    echo "Run: python training/prepare_data.py --output_path ${DATA_PATH}" >&2
    exit 1
fi
if [[ ! -f "${DS_CONFIG}" ]]; then
    echo "ERROR: DeepSpeed config not found at ${DS_CONFIG}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# --- Runtime env ---
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Prevent NCCL P2P hangs on some multi-GPU boxes; comment out if your topology is fine.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

echo "=============================================="
echo "  SafeQwen2.5-VL Video Safety Fine-tuning"
echo "=============================================="
echo "  GPUs             : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "  Model            : ${MODEL_NAME}"
echo "  Data             : ${DATA_PATH}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Epochs           : ${NUM_TRAIN_EPOCHS}"
echo "  Per-device BS    : ${PER_DEVICE_BS}"
echo "  Grad accum       : ${GRAD_ACCUM}"
echo "  Effective BS     : $((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))"
echo "  LR               : ${LR}"
echo "=============================================="

# --- Launch training ---
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    training/train.py \
    --deepspeed "${DS_CONFIG}" \
    --model_name_or_path "${MODEL_NAME}" \
    --processor_name "${PROCESSOR_NAME}" \
    --trust_remote_code True \
    --data_path "${DATA_PATH}" \
    --max_frames 16 \
    --fps 1.0 \
    --max_length 2048 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --safety_head_lr 1e-4 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --logging_dir "${OUTPUT_DIR}/runs" \
    --save_strategy epoch \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training complete. Output saved to ${OUTPUT_DIR}"
