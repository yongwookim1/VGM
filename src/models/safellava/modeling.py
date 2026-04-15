"""Model loading helpers for SafeLLaVA binary safety training."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def load_safellava(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """Load SafeLLaVA and reset its safety head for binary classification."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    model.config.safety_categories = ["safe", "unsafe"]
    model.config.num_safety_categories = 2

    hidden_size = model.config.hidden_size
    safety_head_hidden_scale = getattr(model.config, "safety_head_hidden_scale", 0.5)
    safety_hidden_size = int(hidden_size * safety_head_hidden_scale)
    safety_num_hidden_layers = getattr(model.config, "safety_num_hidden_layers", 1)

    from safellava.model.language_model.safe_llava_llama import SafetyMLP as LLaVASafetyMLP

    model.img_safety_head = LLaVASafetyMLP(
        input_size=hidden_size,
        hidden_size=safety_hidden_size,
        output_size=2,
        safety_num_hidden_layers=safety_num_hidden_layers,
    )
    return model
