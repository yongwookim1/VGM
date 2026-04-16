#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_pipeline.sh --stage prepare
  bash run_pipeline.sh --model safegem --stage check
  bash run_pipeline.sh --model safeqwen --stage train
  bash run_pipeline.sh --model safegem --stage train
  bash run_pipeline.sh --model safellava --stage eval
  bash run_pipeline.sh --model safegem --stage all

Options:
  --model           safegem | safeqwen | safellava | guardreasoner
  --stage           prepare | check | train | eval | all
  --benchmark       safety | mmlu | all (default: all)
  -h, --help        Show this message

Environment variables:
  VIDEOCHATGPT_DIR, SAFETYBENCH_DIR, SAFEWATCH_DIR, SAFEWATCH_MANIFEST
  MODEL_NAME, BASE_MODEL, MODEL_PATH, PROCESSOR_NAME
  SAFELLAVA_PYTHONPATH
  MMLU_PATH, MMLU_SPLIT, MMLU_MAX_SAMPLES
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
BENCHMARK="all"

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
        --benchmark)
            BENCHMARK="$2"
            shift 2
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

case "${BENCHMARK}" in
    safety|mmlu|all) ;;
    *)
        echo "ERROR: unsupported benchmark '${BENCHMARK}'" >&2
        usage >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUTS_DIR="${REPO_ROOT}/outputs"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
DATA_PATH_OVERRIDDEN=false
TEST_DATA_OVERRIDDEN=false
OUTPUT_PATH_OVERRIDDEN=false
if [[ -n "${DATA_PATH+x}" ]]; then
    DATA_PATH_OVERRIDDEN=true
fi
if [[ -n "${TEST_DATA+x}" ]]; then
    TEST_DATA_OVERRIDDEN=true
fi
if [[ -n "${OUTPUT_PATH+x}" ]]; then
    OUTPUT_PATH_OVERRIDDEN=true
fi
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/processed/train_data.json}"
TEST_DATA="${TEST_DATA:-${REPO_ROOT}/data/processed/test_data.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/data/processed/train_data.json}"

mkdir -p "${OUTPUTS_DIR}" "${RESULTS_DIR}"

ensure_model_required() {
    if [[ -z "${MODEL_TYPE}" ]]; then
        echo "ERROR: --model is required for stage '${STAGE}'" >&2
        exit 1
    fi
}

require_path() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} file not found: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local label="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: ${label} directory not found: ${path}" >&2
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
    local exact_path="${OUTPUTS_DIR}/${prefix}"
    if [[ -d "${exact_path}" ]]; then
        printf '%s\n' "${exact_path}"
        return 0
    fi
    local match
    match="$(ls -dt "${OUTPUTS_DIR}/${prefix}"-* 2>/dev/null | head -1 || true)"
    printf '%s\n' "${match}"
}

derive_test_output_path() {
    local output_path="$1"
    local dir
    local filename
    dir="$(dirname "${output_path}")"
    filename="$(basename "${output_path}")"
    if [[ "${filename}" == *train_data* ]]; then
        printf '%s/%s\n' "${dir}" "${filename/train_data/test_data}"
        return 0
    fi

    local stem="${filename%.*}"
    local ext=""
    if [[ "${filename}" == *.* ]]; then
        ext=".${filename##*.}"
    fi
    printf '%s/%s_test%s\n' "${dir}" "${stem}" "${ext}"
}

compute_grad_accum() {
    local global_batch_size="$1"
    local per_device_bs="$2"
    local num_gpus="$3"
    local denom=$((per_device_bs * num_gpus))
    if (( denom <= 0 )); then
        echo "ERROR: invalid batch configuration: per_device_bs=${per_device_bs}, num_gpus=${num_gpus}" >&2
        exit 1
    fi

    local grad_accum=$(((global_batch_size + denom - 1) / denom))
    if (( global_batch_size % denom != 0 )); then
        echo "WARNING: GLOBAL_BATCH_SIZE=${global_batch_size} is not divisible by PER_DEVICE_BS*num_gpus=${denom}; using gradient_accumulation_steps=${grad_accum}." >&2
    fi
    echo "${grad_accum}"
}

ensure_mmlu_path() {
    local mmlu_path="${1}"
    if [[ ! -e "${mmlu_path}" ]]; then
        echo "ERROR: MMLU path not found: ${mmlu_path}" >&2
        echo "Set MMLU_PATH or disable MMLU with --benchmark safety." >&2
        exit 1
    fi
}

ensure_safellava_pythonpath() {
    local root="$1"
    require_dir "SafeLLaVA code path" "${root}"
    require_file "SafeLLaVA module" "${root}/safellava/constants.py"
    require_file "SafeLLaVA module" "${root}/safellava/conversation.py"
    require_file "SafeLLaVA module" "${root}/safellava/mm_utils.py"
}

has_prepare_inputs() {
    [[ -n "${VIDEOCHATGPT_DIR:-}" || -n "${SAFETYBENCH_DIR:-}" || -n "${SAFEWATCH_DIR:-}" || -n "${SAFEWATCH_MANIFEST:-}" ]]
}

maybe_prepare() {
    if has_prepare_inputs; then
        preflight_prepare
        run_prepare
    else
        ensure_data_or_prepare_inputs "Training data" "${DATA_PATH}"
        ensure_data_or_prepare_inputs "Test data" "${TEST_DATA}"
    fi
}

ensure_data_or_prepare_inputs() {
    local label="$1"
    local file_path="$2"
    if [[ -f "${file_path}" ]]; then
        return 0
    fi
    if has_prepare_inputs; then
        return 0
    fi
    echo "ERROR: ${label} not found: ${file_path}" >&2
    echo "Provide processed data or set raw dataset paths for preparation." >&2
    exit 1
}

preflight_prepare() {
    if [[ -z "${VIDEOCHATGPT_DIR:-}" && -z "${SAFETYBENCH_DIR:-}" && -z "${SAFEWATCH_DIR:-}" && -z "${SAFEWATCH_MANIFEST:-}" ]]; then
        echo "ERROR: prepare stage requires at least one data source: VIDEOCHATGPT_DIR, SAFETYBENCH_DIR, SAFEWATCH_DIR, or SAFEWATCH_MANIFEST." >&2
        exit 1
    fi
    if [[ -n "${VIDEOCHATGPT_DIR:-}" ]]; then
        require_dir "VideoChatGPT directory" "${VIDEOCHATGPT_DIR}"
    fi
    if [[ -n "${SAFETYBENCH_DIR:-}" ]]; then
        require_dir "Video-SafetyBench directory" "${SAFETYBENCH_DIR}"
    fi
    if [[ -n "${SAFEWATCH_DIR:-}" ]]; then
        require_dir "SafeWatch directory" "${SAFEWATCH_DIR}"
    fi
    if [[ -n "${SAFEWATCH_MANIFEST:-}" ]]; then
        require_file "SafeWatch manifest" "${SAFEWATCH_MANIFEST}"
    fi
}

preflight_train_safegem() {
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    require_dir "SafeGem model" "${model_name}"
    require_dir "SafeGem processor" "${processor_name}"
    require_file "DeepSpeed config" "${ds_config}"
    ensure_data_or_prepare_inputs "Training data" "${DATA_PATH}"
}

preflight_train_safeqwen() {
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
    local processor_name="${PROCESSOR_NAME:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    require_dir "SafeQwen model" "${model_name}"
    require_dir "SafeQwen processor" "${processor_name}"
    require_file "DeepSpeed config" "${ds_config}"
    ensure_data_or_prepare_inputs "Training data" "${DATA_PATH}"
}

preflight_eval_safegem() {
    local strict_checkpoint="${1:-false}"
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${REPO_ROOT}/models/SafeGem-12B}"
    require_dir "SafeGem base model" "${base_model}"
    require_dir "SafeGem processor" "${processor_name}"
    if [[ "${BENCHMARK}" == "safety" || "${BENCHMARK}" == "all" ]]; then
        ensure_data_or_prepare_inputs "Test data" "${TEST_DATA}"
    fi
    if [[ "${BENCHMARK}" == "mmlu" || "${BENCHMARK}" == "all" ]]; then
        ensure_mmlu_path "${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
    fi
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safegem-video-lora)"
    fi
    if [[ -n "${model_path}" ]]; then
        require_dir "SafeGem checkpoint" "${model_path}"
    elif [[ "${strict_checkpoint}" == "true" ]]; then
        echo "ERROR: SafeGem checkpoint not found under ${OUTPUTS_DIR}" >&2
        exit 1
    else
        echo "NOTE: SafeGem checkpoint not found yet; eval output validation skipped." >&2
    fi
}

preflight_eval_safeqwen() {
    local strict_checkpoint="${1:-false}"
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
    local processor_name="${PROCESSOR_NAME:-${base_model}}"
    require_dir "SafeQwen base model" "${base_model}"
    require_dir "SafeQwen processor" "${processor_name}"
    if [[ "${BENCHMARK}" == "safety" || "${BENCHMARK}" == "all" ]]; then
        ensure_data_or_prepare_inputs "Test data" "${TEST_DATA}"
    fi
    if [[ "${BENCHMARK}" == "mmlu" || "${BENCHMARK}" == "all" ]]; then
        ensure_mmlu_path "${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
    fi
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safeqwen-video-lora)"
    fi
    if [[ -n "${model_path}" ]]; then
        require_dir "SafeQwen checkpoint" "${model_path}"
    elif [[ "${strict_checkpoint}" == "true" ]]; then
        echo "ERROR: SafeQwen checkpoint not found under ${OUTPUTS_DIR}" >&2
        exit 1
    else
        echo "NOTE: SafeQwen checkpoint not found yet; eval output validation skipped." >&2
    fi
}

preflight_train_safellava() {
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeLLaVA-7B}"
    local safellava_pythonpath="${SAFELLAVA_PYTHONPATH:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    require_dir "SafeLLaVA model" "${model_name}"
    ensure_safellava_pythonpath "${safellava_pythonpath}"
    require_file "DeepSpeed config" "${ds_config}"
    ensure_data_or_prepare_inputs "Training data" "${DATA_PATH}"
}

preflight_eval_safellava() {
    local strict_checkpoint="${1:-false}"
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeLLaVA-7B}"
    local safellava_pythonpath="${SAFELLAVA_PYTHONPATH:-${base_model}}"
    require_dir "SafeLLaVA base model" "${base_model}"
    ensure_safellava_pythonpath "${safellava_pythonpath}"
    if [[ "${BENCHMARK}" == "safety" || "${BENCHMARK}" == "all" ]]; then
        ensure_data_or_prepare_inputs "Test data" "${TEST_DATA}"
    fi
    if [[ "${BENCHMARK}" == "mmlu" || "${BENCHMARK}" == "all" ]]; then
        ensure_mmlu_path "${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
    fi
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safellava-video-lora)"
    fi
    if [[ -n "${model_path}" ]]; then
        require_dir "SafeLLaVA checkpoint" "${model_path}"
    elif [[ "${strict_checkpoint}" == "true" ]]; then
        echo "ERROR: SafeLLaVA checkpoint not found under ${OUTPUTS_DIR}" >&2
        exit 1
    else
        echo "NOTE: SafeLLaVA checkpoint not found yet; eval output validation skipped." >&2
    fi
}

preflight_guardreasoner() {
    local model_path="${MODEL_PATH:-${REPO_ROOT}/models/GuardReasoner-VL-3B}"
    require_dir "GuardReasoner model" "${model_path}"
    require_file "Test data" "${TEST_DATA}"
}

run_preflight() {
    case "${MODEL_TYPE}" in
        safegem)
            preflight_train_safegem
            preflight_eval_safegem
            ;;
        safeqwen)
            preflight_train_safeqwen
            preflight_eval_safeqwen
            ;;
        safellava)
            preflight_train_safellava
            preflight_eval_safellava
            ;;
        guardreasoner)
            if [[ "${BENCHMARK}" != "safety" ]]; then
                echo "ERROR: guardreasoner only supports the safety benchmark" >&2
                exit 1
            fi
            preflight_guardreasoner
            ;;
        *)
            echo "ERROR: unsupported model '${MODEL_TYPE}' for check" >&2
            exit 1
            ;;
    esac
    echo "Preflight checks passed for model=${MODEL_TYPE}, benchmark=${BENCHMARK}."
}

run_prepare() {
    local output_path="${OUTPUT_PATH}"
    local args=(--output_path "${output_path}")

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

    if [[ "${DATA_PATH_OVERRIDDEN}" != "true" ]]; then
        DATA_PATH="${output_path}"
    fi
    if [[ "${TEST_DATA_OVERRIDDEN}" != "true" ]]; then
        TEST_DATA="$(derive_test_output_path "${output_path}")"
    fi
}

train_safegem() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    local num_gpus
    num_gpus="$(count_gpus)"
    local master_port="${MASTER_PORT:-29500}"
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    local output_dir="${OUTPUT_DIR:-${OUTPUTS_DIR}/safegem-video-lora}"
    local epochs="${NUM_TRAIN_EPOCHS:-7}"
    local global_batch_size="${GLOBAL_BATCH_SIZE:-128}"
    local per_device_bs="${PER_DEVICE_BS:-2}"
    local grad_accum
    grad_accum="$(compute_grad_accum "${global_batch_size}" "${per_device_bs}" "${num_gpus}")"

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
        --max_length "${MAX_LENGTH:-2048}" \
        --use_lora True \
        --lora_r "${LORA_R:-64}" \
        --lora_alpha "${LORA_ALPHA:-64}" \
        --lora_dropout "${LORA_DROPOUT:-0.05}" \
        --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
        --safety_head_lr "${SAFETY_HEAD_LR:-1e-5}" \
        --output_dir "${output_dir}" \
        --num_train_epochs "${epochs}" \
        --per_device_train_batch_size "${per_device_bs}" \
        --gradient_accumulation_steps "${grad_accum}" \
        --learning_rate "${LR:-1e-5}" \
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
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
        --remove_unused_columns False \
        --ddp_find_unused_parameters False \
        --report_to "${REPORT_TO:-tensorboard}" \
        2>&1 | tee "${output_dir}/train.log"

    LAST_OUTPUT_DIR="${output_dir}"
}

train_safeqwen() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    local num_gpus
    num_gpus="$(count_gpus)"
    local master_port="${MASTER_PORT:-29502}"
    local model_name="${MODEL_NAME:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
    local processor_name="${PROCESSOR_NAME:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    local output_dir="${OUTPUT_DIR:-${OUTPUTS_DIR}/safeqwen-video-lora}"
    local epochs="${NUM_TRAIN_EPOCHS:-5}"
    local global_batch_size="${GLOBAL_BATCH_SIZE:-128}"
    local per_device_bs="${PER_DEVICE_BS:-4}"
    local grad_accum
    grad_accum="$(compute_grad_accum "${global_batch_size}" "${per_device_bs}" "${num_gpus}")"

    mkdir -p "${output_dir}"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

    torchrun \
        --nproc_per_node="${num_gpus}" \
        --master_port="${master_port}" \
        -m src.models.safeqwen.train \
        --deepspeed "${ds_config}" \
        --model_name_or_path "${model_name}" \
        --processor_name "${processor_name}" \
        --trust_remote_code True \
        --data_path "${DATA_PATH}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_length "${MAX_LENGTH:-2048}" \
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
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
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
    local safellava_pythonpath="${SAFELLAVA_PYTHONPATH:-${model_name}}"
    local ds_config="${DS_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
    local output_dir="${OUTPUT_DIR:-${OUTPUTS_DIR}/safellava-video-lora}"
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
    local grad_accum
    grad_accum="$(compute_grad_accum "${global_batch_size}" "${per_device_bs}" "${num_gpus}")"

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
        --safellava_pythonpath "${safellava_pythonpath}" \
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
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
        --remove_unused_columns False \
        --ddp_find_unused_parameters False \
        --report_to "${REPORT_TO:-tensorboard}" \
        2>&1 | tee "${output_dir}/train.log"

    LAST_OUTPUT_DIR="${output_dir}"
}

eval_safegem() {
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeGem-12B}"
    local processor_name="${PROCESSOR_NAME:-${REPO_ROOT}/models/SafeGem-12B}"
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
    local benchmark="${BENCHMARK:-safety}"

    if [[ "${benchmark}" == "safety" || "${benchmark}" == "all" ]]; then
        local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

    python3 -m src.eval.run_inference_safegem \
        --model_path "${model_path}" \
        --base_model "${base_model}" \
        --processor_name "${processor_name}" \
        --test_data "${TEST_DATA}" \
        --output_file "${predictions}" \
        --max_frames "${MAX_FRAMES:-8}" \
        --fps "${FPS:-1.0}" \
        --max_length "${MAX_LENGTH:-8192}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-512}"

        python3 -m src.eval.eval_f1 "${predictions}"
    fi

    if [[ "${benchmark}" == "mmlu" || "${benchmark}" == "all" ]]; then
        local mmlu_path="${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
        local mmlu_predictions="${MMLU_PREDICTIONS:-${RESULTS_DIR}/${run_name}_mmlu_predictions.json}"
        local mmlu_metrics="${MMLU_METRICS:-${RESULTS_DIR}/${run_name}_mmlu_metrics.json}"

        ensure_mmlu_path "${mmlu_path}"
        python3 -m src.eval.run_mmlu \
            --model_type safegem \
            --model_path "${model_path}" \
            --base_model "${base_model}" \
            --mmlu_path "${mmlu_path}" \
            --split "${MMLU_SPLIT:-test}" \
            --output_file "${mmlu_predictions}" \
            --metrics_file "${mmlu_metrics}" \
            --max_new_tokens "${MMLU_MAX_NEW_TOKENS:-8}" \
            --max_samples "${MMLU_MAX_SAMPLES:-0}"
    fi
}

eval_safeqwen() {
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
    local processor_name="${PROCESSOR_NAME:-${base_model}}"
    local model_path="${MODEL_PATH:-}"
    if [[ -z "${model_path}" ]]; then
        model_path="$(latest_checkpoint safeqwen-video-lora)"
    fi
    if [[ -z "${model_path}" ]]; then
        echo "ERROR: No SafeQwen checkpoint found under ${OUTPUTS_DIR}" >&2
        exit 1
    fi

    local run_name
    run_name="$(basename "${model_path}")"
    local benchmark="${BENCHMARK:-all}"

    if [[ "${benchmark}" == "safety" || "${benchmark}" == "all" ]]; then
        local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

        python3 -m src.eval.run_inference_safeqwen \
            --model_path "${model_path}" \
            --base_model "${base_model}" \
            --processor_name "${processor_name}" \
            --test_data "${TEST_DATA}" \
            --output_file "${predictions}" \
            --max_frames "${MAX_FRAMES:-8}" \
            --fps "${FPS:-1.0}" \
            --max_new_tokens "${MAX_NEW_TOKENS:-512}"

        python3 -m src.eval.eval_f1 "${predictions}"
    fi

    if [[ "${benchmark}" == "mmlu" || "${benchmark}" == "all" ]]; then
        local mmlu_path="${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
        local mmlu_predictions="${MMLU_PREDICTIONS:-${RESULTS_DIR}/${run_name}_mmlu_predictions.json}"
        local mmlu_metrics="${MMLU_METRICS:-${RESULTS_DIR}/${run_name}_mmlu_metrics.json}"

        ensure_mmlu_path "${mmlu_path}"
        python3 -m src.eval.run_mmlu \
            --model_type safeqwen \
            --model_path "${model_path}" \
            --base_model "${base_model}" \
            --mmlu_path "${mmlu_path}" \
            --split "${MMLU_SPLIT:-test}" \
            --output_file "${mmlu_predictions}" \
            --metrics_file "${mmlu_metrics}" \
            --max_new_tokens "${MMLU_MAX_NEW_TOKENS:-8}" \
            --max_samples "${MMLU_MAX_SAMPLES:-0}"
    fi
}

eval_safellava() {
    local base_model="${BASE_MODEL:-${REPO_ROOT}/models/SafeLLaVA-7B}"
    local safellava_pythonpath="${SAFELLAVA_PYTHONPATH:-${base_model}}"
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
    local benchmark="${BENCHMARK:-safety}"

    if [[ "${benchmark}" == "safety" || "${benchmark}" == "all" ]]; then
        local predictions="${PREDICTIONS:-${RESULTS_DIR}/${run_name}_predictions.json}"

        python3 -m src.eval.run_inference_safellava \
            --model_path "${model_path}" \
            --base_model "${base_model}" \
            --test_data "${TEST_DATA}" \
            --output_file "${predictions}" \
            --max_frames "${MAX_FRAMES:-8}" \
            --fps "${FPS:-1.0}" \
            --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
            --safellava_pythonpath "${safellava_pythonpath}"

        python3 -m src.eval.eval_f1 "${predictions}"
    fi

    if [[ "${benchmark}" == "mmlu" || "${benchmark}" == "all" ]]; then
        local mmlu_path="${MMLU_PATH:-${REPO_ROOT}/data/mmlu}"
        local mmlu_predictions="${MMLU_PREDICTIONS:-${RESULTS_DIR}/${run_name}_mmlu_predictions.json}"
        local mmlu_metrics="${MMLU_METRICS:-${RESULTS_DIR}/${run_name}_mmlu_metrics.json}"

        ensure_mmlu_path "${mmlu_path}"
        python3 -m src.eval.run_mmlu \
            --model_type safellava \
            --model_path "${model_path}" \
            --base_model "${base_model}" \
            --mmlu_path "${mmlu_path}" \
            --split "${MMLU_SPLIT:-test}" \
            --output_file "${mmlu_predictions}" \
            --metrics_file "${mmlu_metrics}" \
            --max_new_tokens "${MMLU_MAX_NEW_TOKENS:-8}" \
            --max_samples "${MMLU_MAX_SAMPLES:-0}" \
            --safellava_pythonpath "${safellava_pythonpath}"
    fi
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
    preflight_prepare
    run_prepare
    exit 0
fi

ensure_model_required

case "${STAGE}" in
    check)
        run_preflight
        ;;
    train)
        case "${MODEL_TYPE}" in
            safegem)
                preflight_train_safegem
                train_safegem
                ;;
            safeqwen)
                preflight_train_safeqwen
                train_safeqwen
                ;;
            safellava)
                preflight_train_safellava
                train_safellava
                ;;
            *)
                echo "ERROR: training is only supported for safegem or safellava" >&2
                exit 1
                ;;
        esac
        ;;
    eval)
        case "${MODEL_TYPE}" in
            safegem)
                preflight_eval_safegem true
                eval_safegem
                ;;
            safeqwen)
                preflight_eval_safeqwen true
                eval_safeqwen
                ;;
            safellava)
                preflight_eval_safellava true
                eval_safellava
                ;;
            guardreasoner)
                if [[ "${BENCHMARK}" != "safety" ]]; then
                    echo "ERROR: guardreasoner only supports the safety benchmark" >&2
                    exit 1
                fi
                eval_guardreasoner
                ;;
            *)
                echo "ERROR: unsupported model '${MODEL_TYPE}' for eval" >&2
                exit 1
                ;;
        esac
        ;;
    all)
        case "${MODEL_TYPE}" in
            safegem)
                preflight_train_safegem
                preflight_eval_safegem
                maybe_prepare
                train_safegem
                MODEL_PATH="${LAST_OUTPUT_DIR}"
                eval_safegem
                ;;
            safeqwen)
                preflight_train_safeqwen
                preflight_eval_safeqwen
                maybe_prepare
                train_safeqwen
                MODEL_PATH="${LAST_OUTPUT_DIR}"
                eval_safeqwen
                ;;
            safellava)
                preflight_train_safellava
                preflight_eval_safellava
                maybe_prepare
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
