"""
Inference script: run fine-tuned SafeQwen2.5-VL on the test set and write
predictions to a JSON file that is compatible with the downstream eval scripts
(eval_by_string_matching.py, eval_by_openai_parallel_v2.py, etc.).

Usage:
    python eval/run_inference.py \
        --model_path outputs/safeqwen-video-lora-YYYYMMDD_HHMMSS \
        --base_model models/SafeQwen2.5-VL-7B \
        --processor_name Qwen/Qwen2.5-VL-7B-Instruct \
        --test_data training/data/test_data.json \
        --output_file eval/results/predictions.json \
        --device cuda \
        --max_new_tokens 512

The script:
  - Loads the LoRA adapter on top of the base model.
  - Re-uses the same frame sampling / resizing logic as the training dataset.
  - Generates responses with greedy decoding (temperature=0) for reproducibility.
  - Writes the original fields + a "prediction" key to --output_file.
"""

import argparse
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor

# Allow imports from the training package (dataset utilities)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from dataset import sample_frames_from_video, _resize_frame, SYSTEM_PROMPT
from modeling import SafeQwen2_5_VLForConditionalGeneration, SafeQwen2_5_VLConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")


def find_latest_checkpoint():
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    runs = sorted(
        d for d in os.listdir(OUTPUTS_DIR)
        if d.startswith("safeqwen-video-lora-")
        and os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    )
    return os.path.join(OUTPUTS_DIR, runs[-1]) if runs else None


def load_model_and_processor(args):
    logger.info(f"Loading processor from {args.processor_name}")
    processor = AutoProcessor.from_pretrained(
        args.processor_name, trust_remote_code=True
    )

    logger.info(f"Loading base model from {args.base_model}")
    config = SafeQwen2_5_VLConfig.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    model = SafeQwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if not args.no_lora:
        logger.info(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        logger.info("Skipping LoRA — evaluating base model only")

    model.eval()
    model.to(args.device)
    logger.info("Model ready.")
    return model, processor


@torch.inference_mode()
def generate_prediction(model, processor, sample, args):
    """Return (text_response, safety_pred) for a single sample, or (None, None) on error.

    safety_pred is the binary prediction from the safety head (0=safe, 1=unsafe).
    """
    try:
        frames = sample_frames_from_video(
            sample["video_path"],
            max_frames=args.max_frames,
            fps=args.fps,
        )
    except Exception as e:
        logger.warning(f"Cannot load video {sample['video_path']}: {e}")
        return None, None

    frames = [_resize_frame(f) for f in frames]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        videos=[frames],
        padding=False,
        return_tensors="pt",
    ).to(args.device)

    # Forward pass with safety head (no generation needed for classification)
    outputs = model(**inputs, do_safety=True)
    safety_pred = int(outputs.img_safety_logits.argmax(dim=-1).item()) \
        if outputs.img_safety_logits is not None else None

    # Generate text response
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    prediction = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    return prediction, safety_pred


def main():
    parser = argparse.ArgumentParser(description="Run inference on the test set")
    parser.add_argument("--model_path", default=None,
                        help="Path to the LoRA adapter directory. Defaults to latest in outputs/")
    parser.add_argument("--base_model", required=True,
                        help="Path to the base SafeQwen2.5-VL-7B model")
    parser.add_argument("--processor_name", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Processor/tokenizer to use")
    parser.add_argument("--test_data", required=True,
                        help="Path to test_data.json")
    parser.add_argument("--output_file", required=True,
                        help="Where to write predictions JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_lora", action="store_true",
                        help="Evaluate the base model without any LoRA adapter")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples that already have a prediction in --output_file")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = find_latest_checkpoint()
        if args.model_path is None:
            logger.error("No checkpoint found in outputs/. Pass --model_path explicitly.")
            sys.exit(1)
    logger.info(f"Using checkpoint: {args.model_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.test_data) as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples from {args.test_data}")

    # Resume: load any already-generated predictions
    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            results = json.load(f)
        # Only track IDs that are actually present (skip None to avoid false matches)
        done_ids = {r["question_id"] for r in results if r.get("question_id") is not None}
        logger.info(f"Resuming: {len(done_ids)} samples already done")

    model, processor = load_model_and_processor(args)

    for sample in tqdm(test_data, desc="Inference"):
        qid = sample.get("question_id")
        if qid is not None and qid in done_ids:
            continue

        prediction, safety_pred = generate_prediction(model, processor, sample, args)
        result = dict(sample)
        result["prediction"] = prediction if prediction is not None else ""
        result["safety_pred"] = safety_pred  # binary output from safety head
        results.append(result)

        # Save incrementally every 50 samples
        if len(results) % 50 == 0:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    failed = sum(1 for r in results if not r.get("prediction"))
    logger.info(f"Done. {total - failed}/{total} predictions written to {args.output_file}")


if __name__ == "__main__":
    main()
