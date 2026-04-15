"""
Model loading for binary video safety classification.

Supports:
- SafeGem-12B (Gemma3-based): custom SafeGemForConditionalGeneration with binary safety head
- SafeLLaVA-7B (LLaVA/LLaMA+CLIP): loads via trust_remote_code, replaces 20-class head with binary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Gemma3ForConditionalGeneration, Gemma3Config, AutoConfig, AutoModelForCausalLM


@dataclass
class SafeVLOutput(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    img_safety_logits: Optional[torch.FloatTensor] = None
    img_safety_probs: Optional[torch.FloatTensor] = None
    safety_loss: Optional[torch.FloatTensor] = None


class SafetyMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_hidden_layers: int = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.1))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SafeGemConfig(Gemma3Config):
    model_type = "gemma3"

    def __init__(self, safety_categories=None, safety_head_hidden_scale=0.5,
                 safety_loss_lambda=1.0, safety_num_hidden_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.safety_categories = safety_categories or ["safe", "unsafe"]
        self.safety_head_hidden_scale = safety_head_hidden_scale
        self.safety_loss_lambda = safety_loss_lambda
        self.safety_num_hidden_layers = safety_num_hidden_layers
        self.num_safety_categories = len(self.safety_categories)


class SafeGemForConditionalGeneration(Gemma3ForConditionalGeneration):
    config_class = SafeGemConfig

    def __init__(self, config: SafeGemConfig):
        super().__init__(config)
        num_safety_categories = getattr(config, "num_safety_categories", None)
        if num_safety_categories and num_safety_categories > 0:
            hidden_size = config.text_config.hidden_size
            safety_hidden_size = int(
                hidden_size * getattr(config, "safety_head_hidden_scale", 0.5)
            )
            safety_num_hidden_layers = getattr(config, "safety_num_hidden_layers", 1)
            self.img_safety_head = SafetyMLP(
                input_size=hidden_size,
                hidden_size=safety_hidden_size,
                output_size=num_safety_categories,
                num_hidden_layers=safety_num_hidden_layers,
            )
        else:
            self.img_safety_head = None

    def _extract_image_features_pooling(
        self,
        hidden_states: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Extract image features using pooling over vision tower output."""
        if image_hidden_states is not None:
            if len(image_hidden_states.shape) == 3:
                return image_hidden_states.mean(dim=1)
            elif len(image_hidden_states.shape) == 4:
                pooled_per_image = image_hidden_states.mean(dim=2)
                return pooled_per_image.mean(dim=1)
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        do_safety: bool = True,
        safety_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, SafeVLOutput]:
        if do_safety and self.img_safety_head is not None and past_key_values is None:
            output_hidden_states = True
            return_dict = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            return_dict=True,
            **kwargs,
        )

        img_safety_logits = None
        img_safety_probs = None
        safety_loss = None

        is_generation = past_key_values is not None
        has_images = pixel_values is not None

        should_do_safety = (
            do_safety
            and self.img_safety_head is not None
            and has_images
            and not is_generation
        )

        if should_do_safety:
            image_hidden_states = getattr(outputs, "image_hidden_states", None)
            last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None

            visual_features = self._extract_image_features_pooling(
                last_hidden, image_hidden_states
            )
            if visual_features is not None:
                img_safety_logits = self.img_safety_head(visual_features)
                img_safety_probs = torch.softmax(img_safety_logits, dim=-1)

        total_loss = outputs.loss
        if img_safety_logits is not None and safety_labels is not None:
            valid_mask = safety_labels != -100
            if valid_mask.any():
                safety_loss = F.cross_entropy(
                    img_safety_logits[valid_mask],
                    safety_labels[valid_mask],
                )
                safety_lambda = getattr(self.config, "safety_loss_lambda", 1.0)
                if total_loss is not None:
                    total_loss = total_loss + safety_lambda * safety_loss
                else:
                    total_loss = safety_lambda * safety_loss

        if is_generation:
            return outputs

        return SafeVLOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=getattr(outputs, "image_hidden_states", None),
            img_safety_logits=img_safety_logits,
            img_safety_probs=img_safety_probs,
            safety_loss=safety_loss,
        )


# ---------------------------------------------------------------------------
# SafeGem-12B loader
# ---------------------------------------------------------------------------
def load_safegem(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """Load SafeGem-12B and override config for binary safety categories."""
    original_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True,
    )
    config_dict = original_config.to_dict()
    config_dict["safety_categories"] = ["safe", "unsafe"]
    config_dict["num_safety_categories"] = 2
    config = SafeGemConfig.from_dict(config_dict)

    model = SafeGemForConditionalGeneration.from_pretrained(
        model_name_or_path, config=config, torch_dtype=torch_dtype,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    )
    return model


# ---------------------------------------------------------------------------
# SafeLLaVA-7B loader
# ---------------------------------------------------------------------------
def load_safellava(model_name_or_path: str, torch_dtype=torch.bfloat16):
    """Load SafeLLaVA and reinitialize safety head for binary classification.

    The pretrained SafeLLaVA has a 20-class safety head. We replace it with
    a 2-class head (safe/unsafe) and reinitialize the weights.
    Requires the model directory on PYTHONPATH so `import safellava` works.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype,
        trust_remote_code=True, low_cpu_mem_usage=True,
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
