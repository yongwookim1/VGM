#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_pipeline.sh --stage prepare [--prepare-first]
  bash run_pipeline.sh --model safegem --stage train
  bash run_pipeline.sh --model safellava --stage eval
  bash run_pipeline.sh --model safegem --stage all [--prepare-first]

Options:
  --model           safegem | safellava | guardreasoner
  --stage           prepare | train | eval | all
  --prepare-first   Run data preparation before train/all
  -h, --help        Show this message

Environment variables:
  VIDEOCHATGPT_DIR, SAFETYBENCH_DIR, SAFEWATCH_DIR, SAFEWATCH_MANIFEST
  MODEL_NAME, BASE_MODEL, MODEL_PATH, PROCESSOR_NAME
  SAFELLAVA_PYTHONPATH
  CUDA_VISIBLE_DEVICES, MASTER_PORT
  OUTPUT_DIR, RESULTS_DIR, TEST_DATA, DATA_PATH, OUTPUT_PATH
  NUM_TRAIN_EPOCHS, GLOBAL_BATCH_SIZE, PER_DEVICE_BS, LR, SAFETY_HEAD_LR
  LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
  WEIGHT_DECAY, WARMUP_RATIO, LR_SCHEDULER_TYPE
  LOGGING_STEPS, SAVE_STRATEGY, SAVE_TOTAL_LIMIT, DATALOADER_NUM_WORKERS
  MAX_FRAMES, FPS, MAX_LENGTH, MAX_NEW_TOKENS, MAX_TOKENS
EOF
}

MODEL_TYPE=""
STAGE=""
PREPARE_FIRST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --prepare-first)
            PREPARE_FIRST=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${STAGE}" ]]; then
    echo "ERROR: --stage is required" >&2
    usage >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUTS_DIR="${REPO_ROOT}/outputs"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/processed/train_data.json}"
TEST_DATA="${TEST_DATA:-${REPO_ROOT}/data/processed/test_data.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/data/processed/train_data.json}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUTS_DIR}" "${RESULTS_DIR}"

ensure_model_required() {
    if [[ -z "${MODEL_TYPE}" ]]; then
        echo "ERROR: --model is required for stage '${STAGE}'" >&2
        exit 1
    fi
}

count_gpus() {
    local cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    IFS=',' read -ra GPU_ARRAY <<< "${cuda_visible_devices}"
    echo "${#GPU_ARRAY[@]}"
}

latest_checkpoint() {
    local prefix="$1"
    ls -dt "${OUTPUTS_DIR}/${prefix}"-* 2>/dev/null | head -1
}

run_prepare() {
    local args=(--output_path "${OUTPUT_PATH}")

    if [[ -n "${VIDEOCHATGPT_DIR:-}" ]]; then
        args+=(--videochatgpt_dir "${VIDEOCHATGPT_DIR}")
    fi
    if [[ -n "${SAFETYBENCH_DIR:-}" ]]; then
        args+=(--safetybench_dir "${SAFETYBENCH_DIR}")
    fi
    if [[ -n "${SAFEWATCH_DIR:-}" ]]; then
        args+=(--safewatch_dir "${SAFEWATCH_DIR}")
    fi
    if [[ -n "${SAFEWATCH_MANIFEST:-}" ]]; then
        args+=(--safewatch_manifest "${SAFEWATCH_MANIFEST}")
    fi

    python3 -m src.data.prepare_data "${args[@]}"
}

train_safegem() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    local num_gpus
    num_gpus="$(count_gpus)"
    local master_port="${MASTER_PORT:-29500}"
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${REPO_ROOT}/models/gemma-3-12b-it}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    local output_dir="${OUTPUT_DIR:-${OUTPUTS_DIR}/safegem-video-lora-${TIMESTAMP}}"
    local epochs="${NUM_TRAIN_EPOCHS:-7}"
    local global_batch_size="${GLOBAL_BATCH_SIZE:-128}"
    local per_device_bs="${PER_DEVICE_BS:-2}"
    local grad_accum=$((global_batch_size / (per_device_bs * num_gpus)))

    mkdir -p "${output_dir}"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

    torchrun \
        --nproc_per_node="${num_gpus}" \
        --master_port="${master_port}" \
        -m src.models.safegem.train \
        --deepspeed "${ds_config}" \
        --model_name_or_path "${model_name}" \
        --processor_name "${processor_name}" \
        --trust_remote_code True \
        --data_path "${DATA_PATH}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_length "${MAX_LENGTH:-8192}" \
        --use_lora True \
        --lora_r "${LORA_R:-64}" \
        --lora_alpha "${LORA_ALPHA:-64}" \
        --lora_dropout "${LORA_DROPOUT:-0.05}" \
        --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
        --safety_head_lr "${SAFETY_HEAD_LR:-5e-5}" \
        --output_dir "${output_dir}" \
        --num_train_epochs "${epochs}" \
        --per_device_train_batch_size "${per_device_bs}" \
        --gradient_accumulation_steps "${grad_accum}" \
        --learning_rate "${LR:-5e-5}" \
        --weight_decay "${WEIGHT_DECAY:-0.01}" \
        --warmup_ratio "${WARMUP_RATIO:-0.03}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing True \
        --logging_steps "${LOGGING_STEPS:-10}" \
        --logging_dir "${output_dir}/runs" \
        --save_strategy "${SAVE_STRATEGY:-epoch}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
        --remove_unused_columns False \
        --ddp_find_unused_parameters False \
        --report_to "${REPORT_TO:-tensorboard}" \
        2>&1 | tee "${output_dir}/train.log"

    LAST_OUTPUT_DIR="${output_dir}"
}

train_safellava() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    local num_gpus
    num_gpus="$(count_gpus)"
    local master_port="${MASTER_PORT:-29501}"
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeLLaVA-7B}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    local output_dir="${OUTPUT_DIR:-${OUTPUTS_DIR}/safellava-video-lora-${TIMESTAMP}}"
    local default_epochs
    if [[ -n "${NUM_TRAIN_EPOCHS:-}" ]]; then
        default_epochs="${NUM_TRAIN_EPOCHS}"
    elif [[ "$(basename "${model_name}")" == *"13B"* ]]; then
        default_epochs=7
    else
        default_epochs=5
    fi
    local global_batch_size="${GLOBAL_BATCH_SIZE:-128}"
    local per_device_bs="${PER_DEVICE_BS:-2}"
    local grad_accum=$((global_batch_size / (per_device_bs * num_gpus)))

    mkdir -p "${output_dir}"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

    torchrun \
        --nproc_per_node="${num_gpus}" \
        --master_port="${master_port}" \
        -m src.models.safellava.train \
        --deepspeed "${ds_config}" \
        --model_name_or_path "${model_name}" \
        --trust_remote_code True \
        --safellava_pythonpath "${SAFELLAVA_PYTHONPATH:-}" \
        --data_path "${DATA_PATH}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_length "${MAX_LENGTH:-2048}" \
        --use_lora True \
        --lora_r "${LORA_R:-64}" \
        --lora_alpha "${LORA_ALPHA:-64}" \
        --lora_dropout "${LORA_DROPOUT:-0.05}" \
        --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
        --safety_head_lr "${SAFETY_HEAD_LR:-1e-5}" \
        --output_dir "${output_dir}" \
        --num_train_epochs "${default_epochs}" \
        --per_device_train_batch_size "${per_device_bs}" \
        --gradient_accumulation_steps "${grad_accum}" \
        --learning_rate "${LR:-5e-5}" \
        --weight_decay "${WEIGHT_DECAY:-0.01}" \
        --warmup_ratio "${WARMUP_RATIO:-0.03}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing True \
        --logging_steps "${LOGGING_STEPS:-10}" \
        --logging_dir "${output_dir}/runs" \
        --save_strategy "${SAVE_STRATEGY:-epoch}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
        --remove_unused_columns False \
        --ddp_find_unused_parameters False \
        --report_to "${REPORT_TO:-tensorboard}" \
        2>&1 | tee "${output_dir}/train.log"

    LAST_OUTPUT_DIR="${output_dir}"
}

eval_safegem() {
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${REPO_ROOT}/models/gemma-3-12b-it}"
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safegem-video-lora)"
    fi
    if [[ -z "${model_path}" ]]; then
        echo "ERROR: No SafeGem checkpoint found under ${OUTPUTS_DIR}" >&2
        exit 1
    fi

    local run_name
    run_name="$(basename "${model_path}")"
    local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

    python3 -m src.eval.run_inference_safegem \
        --model_path "${model_path}" \
        --base_model "${base_model}" \
        --processor_name "${processor_name}" \
        --test_data "${TEST_DATA}" \
        --output_file "${predictions}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-512}"

    python3 -m src.eval.eval_f1 "${predictions}"
}

eval_safellava() {
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeLLaVA-7B}"
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safellava-video-lora)"
    fi
    if [[ -z "${model_path}" ]]; then
        echo "ERROR: No SafeLLaVA checkpoint found under ${OUTPUTS_DIR}" >&2
        exit 1
    fi

    local run_name
    run_name="$(basename "${model_path}")"
    local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

    python3 -m src.eval.run_inference_safellava \
        --model_path "${model_path}" \
        --base_model "${base_model}" \
        --test_data "${TEST_DATA}" \
        --output_file "${predictions}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
        --safellava_pythonpath "${SAFELLAVA_PYTHONPATH:-}"

    python3 -m src.eval.eval_f1 "${predictions}"
}

eval_guardreasoner() {
    local model_path="${MODEL_PATH:-${REPO_ROOT}/models/GuardReasoner-VL-3B}"
    local run_name
    run_name="$(basename "${model_path}")"
    local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

    python3 -m src.eval.run_inference_guardreasoner \
        --model_path "${model_path}" \
        --test_data "${TEST_DATA}" \
        --output_file "${predictions}" \
        --max_tokens "${MAX_TOKENS:-4096}" \
        --fps "${FPS:-1.0}"

    python3 -m src.eval.eval_f1 "${predictions}"
}

if [[ "${STAGE}" == "prepare" ]]; then
    run_prepare
    exit 0
fi

ensure_model_required

case "${STAGE}" in
    train)
        if [[ "${PREPARE_FIRST}" == "1" ]]; then
            run_prepare
        fi
        case "${MODEL_TYPE}" in
            safegem) train_safegem ;;
            safellava) train_safellava ;;
            *)
                echo "ERROR: training is only supported for safegem or safellava" >&2
                exit 1
                ;;
        esac
        ;;
    eval)
        case "${MODEL_TYPE}" in
            safegem) eval_safegem ;;
            safellava) eval_safellava ;;
            guardreasoner) eval_guardreasoner ;;
            *)
                echo "ERROR: unsupported model '${MODEL_TYPE}' for eval" >&2
                exit 1
                ;;
        esac
        ;;
    all)
        if [[ "${PREPARE_FIRST}" == "1" ]]; then
            run_prepare
        fi
        case "${MODEL_TYPE}" in
            safegem)
                train_safegem
                MODEL_PATH="${LAST_OUTPUT_DIR}"
                eval_safegem
                ;;
            safellava)
                train_safellava
                MODEL_PATH="${LAST_OUTPUT_DIR}"
                eval_safellava
                ;;
            *)
                echo "ERROR: stage 'all' is only supported for safegem or safellava" >&2
                exit 1
                ;;
        esac
        ;;
    *)
        echo "ERROR: unsupported stage '${STAGE}'" >&2
        usage >&2
        exit 1
        ;;
esac
