"""
Main training script for fine-tuning SafeGem-12B on video datasets
with binary safety classification (safe/unsafe).

Supports:
- LoRA via PEFT (attention + MLP projections) with full safety head training
- DeepSpeed ZeRO-2 for multi-GPU
- Joint language modeling + safety classification loss
- Gradient checkpointing for memory efficiency

Usage:
    torchrun --nproc_per_node=4 training/train.py \
        --model_name_or_path etri-vilab/SafeGem-12B \
        --data_path training/data/train_data.json \
        --output_dir outputs/safegem-video-lora
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

# Add parent dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling import SafeGemForConditionalGeneration, SafeGemConfig
from dataset import VideoSafetyDataset
from collator import VideoSafetyCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="etri-vilab/SafeGem-12B",
        metadata={"help": "Path to pretrained model or HuggingFace model ID"},
    )
    processor_name: str = field(
        default="google/gemma-3-12b-it",
        metadata={"help": "Processor/tokenizer to use"},
    )
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to unified train_data.json"},
    )
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
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of LoRA target modules"},
    )
    safety_head_lr: Optional[float] = field(
        default=1e-4,
        metadata={"help": "Separate learning rate for safety head (None = use main LR)"},
    )


class VideoSafetyTrainer(Trainer):
    """Custom trainer that passes safety_labels to the model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        safety_labels = inputs.pop("safety_labels", None)
        outputs = model(
            **inputs,
            do_safety=True,
            safety_labels=safety_labels,
        )
        loss = outputs.loss

        # Log safety loss separately
        if outputs.safety_loss is not None and self.state.global_step % 10 == 0:
            self.log({"safety_loss": outputs.safety_loss.item()})

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Create optimizer with separate LR for safety head."""
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
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs
        )
        return self.optimizer


def load_model(model_args: ModelArguments):
    """Load SafeGem-12B and override config for binary safety categories."""
    from transformers import AutoConfig
    original_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Build our local SafeGemConfig with binary categories
    config_dict = original_config.to_dict()
    config_dict["safety_categories"] = ["safe", "unsafe"]
    config_dict["num_safety_categories"] = 2
    config = SafeGemConfig.from_dict(config_dict)

    model = SafeGemForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    return model


def apply_lora(model, lora_args: LoRAArguments):
    """Apply LoRA adapters to the model."""
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]

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

    # Unfreeze vision encoder (fully fine-tuned per paper)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Load processor
    logger.info(f"Loading processor from {model_args.processor_name}")
    processor = AutoProcessor.from_pretrained(
        model_args.processor_name,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = load_model(model_args)

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Apply LoRA
    if lora_args.use_lora:
        logger.info("Applying LoRA adapters")
        model = apply_lora(model, lora_args)

    # Create dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    train_dataset = VideoSafetyDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_frames=data_args.max_frames,
        fps=data_args.fps,
        max_length=data_args.max_length,
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")

    # Create collator
    collator = VideoSafetyCollator(
        pad_token_id=processor.tokenizer.pad_token_id or 0,
        max_length=data_args.max_length,
    )

    # Create trainer
    trainer = VideoSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )

    # Set safety head LR on trainer
    if lora_args.safety_head_lr is not None:
        trainer._safety_head_lr = lora_args.safety_head_lr

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    # Save fine-tuned visual encoder weights separately
    if lora_args.use_lora:
        visual_state = {
            k: v.cpu()
            for k, v in trainer.model.named_parameters()
            if "vision_tower" in k or "multi_modal_projector" in k
        }
        visual_path = os.path.join(training_args.output_dir, "visual_encoder.bin")
        torch.save(visual_state, visual_path)
        logger.info(f"Saved visual encoder weights ({len(visual_state)} tensors) to {visual_path}")

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
