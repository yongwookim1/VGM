"""Training entrypoint for SafeGem on the unified video safety dataset."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from src.common.collator import SafeGemVideoCollator
from src.models.safegem.dataset import VideoSafetyDataset
from src.models.safegem.modeling import load_safegem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/SafeGem-12B")
    processor_name: str = field(default="models/SafeGem-12B")
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default="data/processed/train_data.json")
    max_frames: int = field(default=8)
    fps: float = field(default=1.0)
    max_length: int = field(default=8192)


@dataclass
class LoRAArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    safety_head_lr: Optional[float] = field(default=5e-5)


class VideoSafetyTrainer(Trainer):
    """Custom trainer that forwards safety labels to the model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        safety_labels = inputs.pop("safety_labels", None)
        outputs = model(**inputs, do_safety=True, safety_labels=safety_labels)
        loss = outputs.loss
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
        if "vision_tower" in name or "multi_modal_projector" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("Loading processor from %s", model_args.processor_name)
    processor = AutoProcessor.from_pretrained(
        model_args.processor_name,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info("Loading model from %s", model_args.model_name_or_path)
    model = load_safegem(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if lora_args.use_lora:
        logger.info("Applying LoRA adapters")
        model = apply_lora(model, lora_args)

    train_dataset = VideoSafetyDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_frames=data_args.max_frames,
        fps=data_args.fps,
        max_length=data_args.max_length,
    )
    collator = SafeGemVideoCollator(
        pad_token_id=processor.tokenizer.pad_token_id or 0,
        max_length=data_args.max_length,
    )

    trainer = VideoSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
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
            if "vision_tower" in key or "multi_modal_projector" in key
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
