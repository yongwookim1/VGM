"""
SafeLLaVA-7B wrapper for binary video safety classification.

Loads SafeLLaVA from HuggingFace (with trust_remote_code) and overrides
the safety head to use binary categories (safe/unsafe) instead of 20.

The model's own safellava/ package handles:
- Vision tower (CLIP ViT-L/14-336px)
- Multimodal projector
- Image embedding into LLaMA hidden states
- Visual token pooling for safety classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


BINARY_SAFETY_CATEGORIES = ["safe", "unsafe"]


def load_safellava_binary(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """Load SafeLLaVA and reinitialize safety head for binary classification.

    The pretrained SafeLLaVA has a 20-class safety head. We replace it with
    a 2-class head (safe/unsafe) and reinitialize the weights.
    """
    # Load the full model with its custom code
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    # Override config for binary categories
    model.config.safety_categories = BINARY_SAFETY_CATEGORIES
    model.config.num_safety_categories = 2

    # Reinitialize safety head for 2 classes (was 20)
    hidden_size = model.config.hidden_size
    safety_head_hidden_scale = getattr(model.config, "safety_head_hidden_scale", 0.5)
    safety_hidden_size = int(hidden_size * safety_head_hidden_scale)
    safety_num_hidden_layers = getattr(model.config, "safety_num_hidden_layers", 1)

    # Import SafetyMLP from the model's own package
    from safellava.model.language_model.safe_llava_llama import SafetyMLP

    model.img_safety_head = SafetyMLP(
        input_size=hidden_size,
        hidden_size=safety_hidden_size,
        output_size=2,
        safety_num_hidden_layers=safety_num_hidden_layers,
    )

    return model
