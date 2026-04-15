#!/bin/bash
# Evaluate GuardReasoner-VL on the same test set used for SafeQwen.
#
# Usage:
#   bash run_eval_guardreasoner.sh
#   bash run_eval_guardreasoner.sh --model_path /path/to/GuardReasoner-VL-3B
#   bash run_eval_guardreasoner.sh --resume

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEST_DATA="${REPO_ROOT}/training/data/test_data.json"
RESULTS_DIR="${REPO_ROOT}/results"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models_cache/GuardReasoner-VL-3B}"
RESUME=""
MAX_TOKENS=4096
FPS=1.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --test_data)    TEST_DATA="$2";  shift 2 ;;
        --max_tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --fps)          FPS="$2";        shift 2 ;;
        --resume)       RESUME="--resume"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Set MODEL_PATH or pass --model_path"
    exit 1
fi

if [[ ! -f "$TEST_DATA" ]]; then
    echo "ERROR: Test data not found at ${TEST_DATA}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
PREDICTIONS="${RESULTS_DIR}/${MODEL_NAME}_predictions.json"

echo "=============================================="
echo "  GuardReasoner-VL Evaluation"
echo "=============================================="
echo "  Model    : ${MODEL_PATH}"
echo "  Test data: ${TEST_DATA}"
echo "  Output   : ${PREDICTIONS}"
echo "=============================================="

# --- Step 1: Inference ---
echo ""
echo "[1/2] Running inference..."
python3 "${REPO_ROOT}/eval/run_inference_guardreasoner.py" \
    --model_path  "${MODEL_PATH}" \
    --test_data   "${TEST_DATA}" \
    --output_file "${PREDICTIONS}" \
    --max_tokens  "${MAX_TOKENS}" \
    --fps         "${FPS}" \
    ${RESUME}

# --- Step 2: F1 evaluation ---
echo ""
echo "[2/2] Computing F1..."
python3 "${REPO_ROOT}/eval/eval_f1.py" "${PREDICTIONS}"
