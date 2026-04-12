#!/bin/bash
# Training script for SafeQwen2.5-VL-7B video fine-tuning on HoliSafe safety categories.
# Hyperparameters follow the original HoliSafe paper (Lee et al., 2025).
#
# Prerequisites:
#   1. pip install transformers peft deepspeed accelerate qwen-vl-utils av opencv-python
#   2. Download ActivityNet videos to the path used by prepare_data.py
#   3. Download & extract Video-SafetyBench videos (video.tar.gz)
#   4. Run: python training/prepare_data.py --output_path training/data/train_data.json
#
# Usage:
#   bash training/scripts/run_train.sh

set -euo pipefail

# --- Configuration ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=4
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
PROCESSOR_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="${REPO_ROOT}/training/data/train_data.json"
DS_CONFIG="${REPO_ROOT}/training/configs/deepspeed_zero2.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/safeqwen-video-lora-$(date +%Y%m%d_%H%M%S)"

# Paper settings (Table 6, SafeQwen column)
NUM_TRAIN_EPOCHS=5
GLOBAL_BATCH_SIZE=128
PER_DEVICE_BS=4
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_GPUS)))  # 128 / (4*4) = 8
LR="5e-5"
SAFETY_HEAD_LR="5e-5"
LORA_R=64
LORA_ALPHA=64

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
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Uncomment for verbose NCCL diagnostics:
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

echo "=============================================="
echo "  SafeQwen2.5-VL Video Safety Fine-tuning"
echo "  (Paper-aligned hyperparameters)"
echo "=============================================="
echo "  GPUs             : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "  Model            : ${MODEL_NAME}"
echo "  Data             : ${DATA_PATH}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Epochs           : ${NUM_TRAIN_EPOCHS}"
echo "  Per-device BS    : ${PER_DEVICE_BS}"
echo "  Grad accum       : ${GRAD_ACCUM}"
echo "  Global BS        : ${GLOBAL_BATCH_SIZE}"
echo "  Backbone LR      : ${LR}"
echo "  Safety head LR   : ${SAFETY_HEAD_LR}"
echo "  LoRA r           : ${LORA_R}"
echo "  LoRA alpha       : ${LORA_ALPHA}"
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
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --safety_head_lr "${SAFETY_HEAD_LR}" \
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
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training complete. Output saved to ${OUTPUT_DIR}"
