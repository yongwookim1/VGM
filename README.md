# VGM

Video safety training and evaluation code for two model backends on one shared
processed dataset schema:

- `SafeGem`
- `SafeLLaVA`

The repo now uses a `src/` package layout so data preparation, shared utilities,
model backends, and evaluation logic are separated cleanly.

## Repo Layout

```text
src/
  data/                 Shared data prep, labels, and video utilities
  common/               Shared collators and IO helpers
  models/safegem/       SafeGem dataset, modeling, training
  models/safellava/     SafeLLaVA dataset, modeling, training
  eval/                 Inference and metrics
scripts/                Thin shell entrypoints
tests/                  Small smoke tests for shared logic
```

## Data Schema

Both backends train on the same processed JSON sample format:

```json
{
  "dataset": "videosafetybench",
  "split": "harmful",
  "question_id": "sample-1",
  "video_path": "/abs/or/relative/path.mp4",
  "question": "Describe the video",
  "answer": "I cannot help with harmful content.",
  "safety_label": 1,
  "category": "violence",
  "subcategory": "physical_attack"
}
```

Binary label contract:

- `0 = safe`
- `1 = unsafe`

## Data Preparation

The unified data-prep entrypoint can combine any subset of:

- `VideoChatGPT`
- `Video-SafetyBench`
- `SafeWatch`

Examples:

```bash
VIDEOCHATGPT_DIR=/data/videochatgpt \
SAFETYBENCH_DIR=/data/video_safetybench \
bash scripts/prepare_data.sh
```

```bash
SAFEWATCH_DIR=/data/SafeWatch \
bash scripts/prepare_data.sh
```

```bash
SAFEWATCH_MANIFEST=/data/SafeWatch/train.json \
bash scripts/prepare_data.sh
```

Processed files are written to:

- `data/processed/train_data.json`
- `data/processed/test_data.json`

## Pipeline

All shell orchestration now goes through one file:

```bash
bash run_pipeline.sh --model safegem --stage train
```

Supported stages:

- `prepare`
- `train`
- `eval`
- `all`

Supported models:

- `safegem`
- `safellava`
- `guardreasoner` (`eval` only)

### Prepare Data

```bash
SAFEWATCH_DIR=./data/SafeWatch \
bash run_pipeline.sh --stage prepare
```

### Train SafeGem

```bash
MODEL_NAME=./models/SafeGem-12B \
PROCESSOR_NAME=./models/gemma-3-12b-it \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_pipeline.sh --model safegem --stage train
```

### Evaluate SafeGem

```bash
MODEL_PATH=./outputs/safegem-video-lora-YYYYMMDD_HHMMSS \
BASE_MODEL=./models/SafeGem-12B \
PROCESSOR_NAME=./models/gemma-3-12b-it \
bash run_pipeline.sh --model safegem --stage eval
```

### Train + Evaluate SafeGem

```bash
MODEL_NAME=./models/SafeGem-12B \
PROCESSOR_NAME=./models/gemma-3-12b-it \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_pipeline.sh --model safegem --stage all
```

### Train + Evaluate SafeLLaVA

```bash
MODEL_NAME=./models/SafeLLaVA-7B \
SAFELLAVA_PYTHONPATH=/path/to/SafeLLaVA/repo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_pipeline.sh --model safellava --stage all
```

### Evaluate GuardReasoner

```bash
MODEL_PATH=./models/GuardReasoner-VL-3B \
bash run_pipeline.sh --model guardreasoner --stage eval
```

## Notes

- The repo only changes code and project structure. Models, videos, and other
  large assets are expected to be downloaded separately on the target server.
- `visual_encoder.bin` is now loaded during evaluation when present, so eval can
  reflect the vision-tower updates saved during training.
- Old `training/` and `eval/` Python files remain as compatibility shims to the
  new `src/` modules.
- If you want data prep before `train` or `all`, add `--prepare-first`.
