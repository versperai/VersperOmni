"""
Vision-Language Model (MiniMind-V) — extends MiniMind with SigLIP2 vision encoder.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .config import VLMConfig
from .model import MiniMindForCausalLM
from .modules.norm import RMSNorm
from .modules.rope import precompute_freqs_cis
from .modules.feed_forward import MOEFeedForward


class MMVisionProjector(nn.Module):
    """MLP projector: encoder hidden → LLM hidden space."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MiniMindVLM(MiniMindForCausalLM):
    """MiniMind with visual input support (SigLIP2 encoder + MLP projector)."""

    config_class = VLMConfig

    def __init__(
        self,
        config: VLMConfig = None,
        vision_model_path="./model/siglip2-base-p32-256-ve",
    ):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self._load_vision(vision_model_path)
        self.vision_proj = MMVisionProjector(
            self.config.image_hidden_size, self.config.hidden_size
        )

    @staticmethod
    def _load_vision(model_path):
        """Load and freeze SigLIP2 vision encoder."""
        if not model_path or not os.path.exists(model_path):
            return None, None
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ["RGBA", "LA"]:
            image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, "keys"):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def _inject_vision_features(self, tokens, h, vision_tensors, seqlen):
        """Replace image placeholder tokens with projected vision features."""
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker = self.config.image_ids[0]
        vf = vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb = h[b]
            seq = tokens[b].tolist()
            k = 0
            i = 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        inject = vf[b][k][: i - start]
                        hb = torch.cat((hb[:start], inject, hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # Inject vision features at the first forward pass
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, "keys"):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    img_emb = MiniMindVLM.get_image_embeddings(
                        {k: v.flatten(0, 1) for k, v in pixel_values.items()},
                        self.vision_encoder,
                    )
                    vision_tensors = (
                        self.vision_proj(img_emb)
                        .view(bs, num, self.config.image_token_len, -1)
                    )
                else:
                    vision_tensors = self.vision_proj(
                        MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
                    )
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack(
                    [
                        self.vision_proj(
                            MiniMindVLM.get_image_embeddings(
                                pixel_values[:, i, :, :, :], self.vision_encoder
                            )
                        )
                        for i in range(num)
                    ],
                    dim=1,
                )
            hidden_states = self._inject_vision_features(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1],
            )

        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.model.freqs_cos = freqs_cos.to(hidden_states.device)
            self.model.freqs_sin = freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos : start_pos + seq_length],
            self.model.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum(
            l.mlp.aux_loss
            for l in self.model.layers
            if isinstance(l.mlp, MOEFeedForward)
        ).to(hidden_states.device) if any(
            isinstance(l.mlp, MOEFeedForward) for l in self.model.layers
        ) else hidden_states.new_zeros(1).squeeze()
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # DDP dummy

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and "pixel_values" in kwargs:
            pv = kwargs["pixel_values"]
            if hasattr(pv, "keys"):
                kwargs["pixel_values"] = {
                    k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1)))
                    for k, v in pv.items()
                }
            else:
                kwargs["pixel_values"] = pv.repeat(
                    num_return_sequences, *([1] * (pv.ndim - 1))
                )
        return super().generate(
            *args, num_return_sequences=num_return_sequences, **kwargs
        )
