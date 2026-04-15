"""
Main training script for fine-tuning SafeLLaVA-7B on video datasets
with binary safety classification (safe/unsafe).

Supports:
- LoRA via PEFT (LLM layers) with full safety head training
- DeepSpeed ZeRO-2 for multi-GPU
- Joint language modeling + safety classification loss
- Gradient checkpointing for memory efficiency

Usage:
    torchrun --nproc_per_node=4 training/train.py \
        --model_name_or_path models/SafeLLaVA-7B \
        --data_path training/data/train_data.json \
        --output_dir outputs/safellava-video-lora
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="etri-vilab/SafeLLaVA-7B",
        metadata={"help": "Path to pretrained model or HuggingFace model ID"},
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
    """Custom trainer that handles SafeLLaVA's forward signature and safety loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        safety_labels = inputs.pop("safety_labels", None)
        image_sizes = inputs.pop("image_sizes", None)

        # SafeLLaVA forward signature: images, image_sizes, do_safety
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            images=inputs.get("images"),
            image_sizes=image_sizes,
            do_safety=True,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=inputs["input_ids"].device)

        # Add safety classification loss
        safety_loss = None
        if outputs.img_safety_logits is not None and safety_labels is not None:
            valid_mask = safety_labels != -100
            if valid_mask.any():
                safety_loss = F.cross_entropy(
                    outputs.img_safety_logits[valid_mask],
                    safety_labels[valid_mask],
                )
                safety_lambda = getattr(model.config, "safety_loss_lambda", 1.0)
                loss = loss + safety_lambda * safety_loss

        if safety_loss is not None and self.state.global_step % 10 == 0:
            self.log({"safety_loss": safety_loss.item()})

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


def setup_safellava_imports(model_name_or_path: str):
    """Add SafeLLaVA's bundled package to Python path."""
    # If it's a local path, add it so `import safellava` works
    model_dir = os.path.abspath(model_name_or_path)
    if os.path.isdir(model_dir):
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        logger.info(f"Added {model_dir} to Python path for safellava imports")


def main():
    # Add training dir to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Setup safellava imports from model directory
    setup_safellava_imports(model_args.model_name_or_path)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with binary safety head
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    from modeling import load_safellava_binary
    model = load_safellava_binary(model_args.model_name_or_path)

    # Get image processor from the vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Apply LoRA
    if lora_args.use_lora:
        logger.info("Applying LoRA adapters")
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

        # Unfreeze vision tower (per paper)
        if getattr(model.config, "unfreeze_mm_vision_tower", True):
            for name, param in model.named_parameters():
                if "vision_tower" in name or "mm_projector" in name:
                    param.requires_grad = True

        model.print_trainable_parameters()

    # Create dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    from dataset import VideoSafetyDataset
    train_dataset = VideoSafetyDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        max_frames=data_args.max_frames,
        fps=data_args.fps,
        max_length=data_args.max_length,
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")

    # Create collator
    from collator import VideoSafetyCollator
    collator = VideoSafetyCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_length=data_args.max_length,
    )

    # Create trainer
    trainer = VideoSafetyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    if lora_args.safety_head_lr is not None:
        trainer._safety_head_lr = lora_args.safety_head_lr

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()

    # Save visual encoder weights separately
    if lora_args.use_lora:
        visual_state = {
            k: v.cpu()
            for k, v in trainer.model.named_parameters()
            if "vision_tower" in k or "mm_projector" in k
        }
        visual_path = os.path.join(training_args.output_dir, "visual_encoder.bin")
        torch.save(visual_state, visual_path)
        logger.info(f"Saved visual encoder weights ({len(visual_state)} tensors) to {visual_path}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
