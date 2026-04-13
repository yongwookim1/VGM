#!/bin/bash
# Evaluate the most recently fine-tuned model on the test set.
#
# Usage:
#   bash eval/run_eval.sh
#   bash eval/run_eval.sh --model_path outputs/safeqwen-video-lora-20260413_120000
#   bash eval/run_eval.sh --resume   # skip already-predicted samples

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/models/SafeQwen2.5-VL-7B}"
OUTPUTS_DIR="${REPO_ROOT}/outputs"
TEST_DATA="${REPO_ROOT}/training/data/test_data.json"
RESULTS_DIR="${REPO_ROOT}/results"

MODEL_PATH=""
RESUME=""
NO_LORA=""
MAX_FRAMES=16
FPS=1.0
MAX_NEW_TOKENS=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)     MODEL_PATH="$2"; shift 2 ;;
        --test_data)      TEST_DATA="$2";  shift 2 ;;
        --max_frames)     MAX_FRAMES="$2"; shift 2 ;;
        --fps)            FPS="$2";        shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --no_lora)        NO_LORA="--no_lora"; shift ;;
        --resume)         RESUME="--resume"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Resolve latest checkpoint ---
if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH=$(ls -dt "${OUTPUTS_DIR}"/safeqwen-video-lora-* 2>/dev/null | head -1)
    if [[ -z "$MODEL_PATH" ]]; then
        echo "ERROR: No checkpoint found under ${OUTPUTS_DIR}. Pass --model_path explicitly."
        exit 1
    fi
fi

if [[ ! -f "$TEST_DATA" ]]; then
    echo "ERROR: Test data not found at ${TEST_DATA}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
if [[ -n "$NO_LORA" ]]; then
    RUN_NAME="safeqwen-base"
else
    RUN_NAME="$(basename "${MODEL_PATH}")"
fi
PREDICTIONS="${RESULTS_DIR}/${RUN_NAME}_predictions.json"

echo "=============================================="
echo "  Evaluation"
echo "=============================================="
echo "  Checkpoint : ${MODEL_PATH}"
echo "  Test data  : ${TEST_DATA}"
echo "  Output     : ${PREDICTIONS}"
echo "=============================================="

# --- Step 1: Inference ---
echo ""
echo "[1/2] Running inference..."
python3 "${REPO_ROOT}/eval/run_inference.py" \
    --model_path    "${MODEL_PATH}" \
    --base_model    "${BASE_MODEL}" \
    --test_data     "${TEST_DATA}" \
    --output_file   "${PREDICTIONS}" \
    --max_frames    "${MAX_FRAMES}" \
    --fps           "${FPS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${NO_LORA} ${RESUME}

# --- Step 2: F1 evaluation ---
echo ""
echo "[2/2] Computing F1..."
python3 "${REPO_ROOT}/eval/eval_f1.py" "${PREDICTIONS}"
