"""Run SafeLLaVA inference on the processed test split."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

from src.common.io import load_json, write_json
from src.data.video_utils import sample_frames_from_video, select_representative_frame
from src.models.safellava.modeling import load_safellava

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


def load_model_components(args):
    if args.safellava_pythonpath:
        sys.path.insert(0, args.safellava_pythonpath)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model = load_safellava(args.base_model)

    if not args.no_lora:
        logger.info("Loading LoRA adapter from %s", args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)
        load_visual_encoder_weights(model, args.model_path)
        model = model.merge_and_unload()
    else:
        load_visual_encoder_weights(model, args.model_path)

    model.eval()
    model.to(args.device)

    from safellava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from safellava.conversation import conv_templates
    from safellava.mm_utils import process_images, tokenizer_image_token

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "model_config": model_config,
        "default_image_token": DEFAULT_IMAGE_TOKEN,
        "image_token_index": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
    }


@torch.inference_mode()
def generate_prediction(components, sample, args):
    model = components["model"]
    tokenizer = components["tokenizer"]
    try:
        frames = sample_frames_from_video(sample["video_path"], max_frames=args.max_frames, fps=args.fps)
    except Exception as exc:
        logger.warning("Cannot load video %s: %s", sample["video_path"], exc)
        return None, None

    image = select_representative_frame(frames)
    image_tensor = components["process_images"](
        [image],
        components["image_processor"],
        components["model_config"],
    )
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    conv = components["conv_templates"]["llava_v1"].copy()
    question = components["default_image_token"] + "\n" + sample["question"]
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = components["tokenizer_image_token"](
        prompt,
        tokenizer,
        components["image_token_index"],
        return_tensors="pt",
    )
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    images = image_tensor.unsqueeze(0).to(args.device)
    image_sizes = [image.size]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        image_sizes=image_sizes,
        do_safety=True,
    )
    safety_pred = (
        int(outputs.img_safety_logits.argmax(dim=-1).item())
        if getattr(outputs, "img_safety_logits", None) is not None
        else None
    )

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        image_sizes=image_sizes,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0][prompt_len:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction, safety_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeLLaVA inference on the test set")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="models/SafeLLaVA-7B")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--safellava_pythonpath", default="")
    args = parser.parse_args()

    test_data = load_json(args.test_data)
    results = []
    done_ids = set()
    if args.resume and os.path.exists(args.output_file):
        results = load_json(args.output_file)
        done_ids = {row["question_id"] for row in results if row.get("question_id") is not None}

    components = load_model_components(args)
    for sample in tqdm(test_data, desc="SafeLLaVA inference"):
        question_id = sample.get("question_id")
        if question_id is not None and question_id in done_ids:
            continue

        prediction, safety_pred = generate_prediction(components, sample, args)
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
