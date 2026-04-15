"""Training entrypoint for SafeLLaVA on the unified video safety dataset."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from src.common.collator import SafeLLaVAVideoCollator
from src.models.safellava.dataset import VideoSafetyDataset
from src.models.safellava.modeling import load_safellava

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/SafeLLaVA-7B")
    trust_remote_code: bool = field(default=True)
    safellava_pythonpath: str = field(
        default="",
        metadata={"help": "Optional path that exposes the local safellava package."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="data/processed/train_data.json")
    max_frames: int = field(default=8)
    fps: float = field(default=1.0)
    max_length: int = field(default=2048)


@dataclass
class LoRAArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    safety_head_lr: Optional[float] = field(default=1e-5)


class VideoSafetyTrainer(Trainer):
    """Custom trainer that forwards safety labels to the model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        safety_labels = inputs.pop("safety_labels", None)
        outputs = model(**inputs, do_safety=True, safety_labels=safety_labels)
        loss = outputs.loss
        if getattr(outputs, "safety_loss", None) is not None and self.state.global_step % 10 == 0:
            self.log({"safety_loss": outputs.safety_loss.item()})
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        safety_head_lr = getattr(self, "_safety_head_lr", None)
        if safety_head_lr is None:
            return super().create_optimizer()

        safety_head_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "img_safety_head" in name:
                safety_head_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": safety_head_params,
                "lr": safety_head_lr,
                "weight_decay": self.args.weight_decay,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, self.model
        )
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def apply_lora(model, lora_args: LoRAArguments):
    target_modules = [module.strip() for module in lora_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["img_safety_head"],
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "vision_tower" in name or "mm_projector" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.safellava_pythonpath:
        sys.path.insert(0, model_args.safellava_pythonpath)

    logger.info("Loading tokenizer/image processor from %s", model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info("Loading model from %s", model_args.model_name_or_path)
    model = load_safellava(model_args.model_name_or_path)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if lora_args.use_lora:
        logger.info("Applying LoRA adapters")
        model = apply_lora(model, lora_args)

    train_dataset = VideoSafetyDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model_config,
        max_frames=data_args.max_frames,
        fps=data_args.fps,
        max_length=data_args.max_length,
    )
    collator = SafeLLaVAVideoCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_length=data_args.max_length,
    )

    trainer = VideoSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if lora_args.safety_head_lr is not None:
        trainer._safety_head_lr = lora_args.safety_head_lr

    logger.info("Starting training")
    train_result = trainer.train()

    logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    trainer.save_state()

    if lora_args.use_lora:
        visual_state = {
            key: value.cpu()
            for key, value in trainer.model.named_parameters()
            if "vision_tower" in key or "mm_projector" in key
        }
        visual_path = os.path.join(training_args.output_dir, "visual_encoder.bin")
        torch.save(visual_state, visual_path)
        logger.info("Saved visual encoder weights to %s", visual_path)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
