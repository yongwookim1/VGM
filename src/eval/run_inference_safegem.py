"""Run SafeGem inference on the processed test split."""

from __future__ import annotations

import argparse
import logging
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor

from src.common.io import load_json, write_json
from src.data.video_utils import sample_frames_from_video
from src.models.safegem.modeling import load_safegem
from src.models.safegem.preprocess import prepare_safegem_inputs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_visual_encoder_weights(model, checkpoint_dir: str) -> None:
    visual_path = os.path.join(checkpoint_dir, "visual_encoder.bin")
    if not os.path.isfile(visual_path):
        return

    logger.info("Loading visual encoder weights from %s", visual_path)
    state_dict = torch.load(visual_path, map_location="cpu")
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        candidate_keys = [key]
        if key.startswith("base_model.model."):
            candidate_keys.append(key.replace("base_model.model.", "", 1))

        for candidate in candidate_keys:
            if candidate in model_state:
                filtered[candidate] = value.to(model_state[candidate].dtype)
                break

    if filtered:
        model.load_state_dict(filtered, strict=False)


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.processor_name, trust_remote_code=True)
    model = load_safegem(args.base_model, trust_remote_code=True)

    if not args.no_lora:
        logger.info("Loading LoRA adapter from %s", args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)
        load_visual_encoder_weights(model, args.model_path)
        model = model.merge_and_unload()
    else:
        load_visual_encoder_weights(model, args.model_path)

    model.eval()
    model.to(args.device)
    return model, processor


@torch.inference_mode()
def generate_prediction(model, processor, sample, args):
    try:
        frames = sample_frames_from_video(
            sample["video_path"],
            max_frames=args.max_frames,
            fps=args.fps,
        )
    except Exception as exc:
        logger.warning("Cannot load video %s: %s", sample["video_path"], exc)
        return None, None

    try:
        frames, inputs, _, _ = prepare_safegem_inputs(
            processor,
            sample,
            frames,
            max_length=args.max_length,
            include_answer=False,
        )
    except Exception as exc:
        logger.warning("Cannot tokenize sample %s: %s", sample.get("question_id"), exc)
        return None, None
    inputs = inputs.to(args.device)

    outputs = model(
        **inputs,
        do_safety=True,
        num_frames_per_sample=torch.tensor([len(frames)], device=args.device),
    )
    safety_pred = (
        int(outputs.img_safety_logits.argmax(dim=-1).item())
        if getattr(outputs, "img_safety_logits", None) is not None
        else None
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction, safety_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeGem inference on the test set")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="models/SafeGem-12B")
    parser.add_argument("--processor_name", default="models/gemma-3-12b-it")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    test_data = load_json(args.test_data)
    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output_file):
        results = load_json(args.output_file)
        done_ids = {row["question_id"] for row in results if row.get("question_id") is not None}

    model, processor = load_model_and_processor(args)
    for sample in tqdm(test_data, desc="SafeGem inference"):
        question_id = sample.get("question_id")
        if question_id is not None and question_id in done_ids:
            continue

        prediction, safety_pred = generate_prediction(model, processor, sample, args)
        result = dict(sample)
        result["prediction"] = prediction or ""
        result["safety_pred"] = safety_pred
        results.append(result)

        if len(results) % 50 == 0:
            write_json(args.output_file, results)

    write_json(args.output_file, results)
    logger.info("Wrote %s predictions to %s", len(results), args.output_file)


if __name__ == "__main__":
    main()
