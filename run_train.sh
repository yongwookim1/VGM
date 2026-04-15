#!/bin/bash
# Training script for SafeGem-12B video fine-tuning with binary safety classification.
# Based on HoliSafe paper hyperparameters, adapted for Gemma3-based SafeGem.
#
# Prerequisites:
#   1. pip install transformers peft deepspeed accelerate opencv-python pillow
#   2. git clone https://huggingface.co/etri-vilab/SafeGem-12B models/SafeGem-12B
#   3. Download ActivityNet videos & Video-SafetyBench videos
#   4. Run: python training/prepare_data.py --output_path training/data/train_data.json
#
# Usage:
#   MODEL_NAME=models/SafeGem-12B bash run_train.sh

set -euo pipefail

# --- Configuration ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra _GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#_GPU_ARRAY[@]}
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="${MODEL_NAME:-etri-vilab/SafeGem-12B}"
PROCESSOR_NAME="${PROCESSOR_NAME:-google/gemma-3-12b-it}"
DATA_PATH="${REPO_ROOT}/training/data/train_data.json"
DS_CONFIG="${REPO_ROOT}/training/configs/deepspeed_zero2.json"
OUTPUT_DIR="${REPO_ROOT}/outputs/safegem-video-lora-$(date +%Y%m%d_%H%M%S)"

# Paper settings (adapted for SafeGem-12B)
NUM_TRAIN_EPOCHS=5
GLOBAL_BATCH_SIZE=128
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_GPUS)))
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

echo "=============================================="
echo "  SafeGem-12B Video Safety Fine-tuning"
echo "  (Binary classification: safe/unsafe)"
echo "=============================================="
echo "  GPUs             : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "  Model            : ${MODEL_NAME}"
echo "  Processor        : ${PROCESSOR_NAME}"
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
    --max_frames 8 \
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
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training complete. Output saved to ${OUTPUT_DIR}"
