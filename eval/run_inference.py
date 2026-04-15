"""
Inference script: run fine-tuned SafeLLaVA-7B on the test set and write
predictions to a JSON file.

Usage:
    python eval/run_inference.py \
        --model_path outputs/safellava-video-lora-YYYYMMDD_HHMMSS \
        --base_model models/SafeLLaVA-7B \
        --test_data training/data/test_data.json \
        --output_file eval/results/predictions.json
"""

import argparse
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

# Allow imports from the training package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")


def find_latest_checkpoint():
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    runs = sorted(
        d for d in os.listdir(OUTPUTS_DIR)
        if d.startswith("safellava-video-lora-")
        and os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    )
    return os.path.join(OUTPUTS_DIR, runs[-1]) if runs else None


def load_model_and_tokenizer(args):
    # Add model dir to path for safellava imports
    model_dir = os.path.abspath(args.base_model)
    if os.path.isdir(model_dir) and model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from modeling import load_safellava_binary

    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True
    )

    logger.info(f"Loading base model from {args.base_model}")
    model = load_safellava_binary(args.base_model)

    # Load vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if not args.no_lora:
        logger.info(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()

    model.eval()
    model.to(args.device)
    logger.info("Model ready.")
    return model, tokenizer, image_processor


@torch.inference_mode()
def generate_prediction(model, tokenizer, image_processor, sample, args):
    from dataset import sample_frames_from_video, select_representative_frame
    from safellava.mm_utils import process_images, tokenizer_image_token
    from safellava.conversation import conv_templates
    from safellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    try:
        frames = sample_frames_from_video(
            sample["video_path"], max_frames=args.max_frames, fps=args.fps
        )
    except Exception as e:
        logger.warning(f"Cannot load video {sample['video_path']}: {e}")
        return None, None

    image = select_representative_frame(frames)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(args.device, dtype=torch.bfloat16)

    conv = conv_templates["llava_v1"].copy()
    question = DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)

    # Forward pass for safety classification
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_safety=True,
        output_hidden_states=True,
        return_dict=True,
    )
    safety_pred = int(outputs.img_safety_logits.argmax(dim=-1).item()) \
        if outputs.img_safety_logits is not None else None

    # Generate text response
    output_ids = model.generate(
        inputs=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0][prompt_len:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction, safety_pred


def main():
    parser = argparse.ArgumentParser(description="Run inference on the test set")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--base_model", default="models/SafeLLaVA-7B")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = find_latest_checkpoint()
        if args.model_path is None:
            logger.error("No checkpoint found. Pass --model_path explicitly.")
            sys.exit(1)
    logger.info(f"Using checkpoint: {args.model_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.test_data) as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")

    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            results = json.load(f)
        done_ids = {r["question_id"] for r in results if r.get("question_id") is not None}
        logger.info(f"Resuming: {len(done_ids)} samples already done")

    model, tokenizer, image_processor = load_model_and_tokenizer(args)

    for sample in tqdm(test_data, desc="Inference"):
        qid = sample.get("question_id")
        if qid is not None and qid in done_ids:
            continue

        prediction, safety_pred = generate_prediction(
            model, tokenizer, image_processor, sample, args
        )
        result = dict(sample)
        result["prediction"] = prediction if prediction is not None else ""
        result["safety_pred"] = safety_pred
        results.append(result)

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
