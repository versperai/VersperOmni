"""
Omni Model (Versper-O) — Thinker-Talker architecture with audio+vision+speech I/O.
"""
import os
import math
import io
import contextlib
import logging
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from types import SimpleNamespace
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import SiglipImageProcessor, SiglipVisionModel

from config import OmniConfig
from models.llm import VersperForCausalLM
from modules.norm import RMSNorm
from modules.rope import precompute_freqs_cis
from modules.block import VersperBlock
from modules.feed_forward import MOEFeedForward


# ═══════════════════════════════════════════════
# Projector modules
# ═══════════════════════════════════════════════

class MMAudioProjector(nn.Module):
    """MLP projector: SenseVoice features → LLM hidden space."""

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


class MMVisionProjector(nn.Module):
    """MLP projector: SigLIP2 features → LLM hidden space."""

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


class TalkerHead(nn.Module):
    """Low-rank output head for 8 Mimi codebooks."""

    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, out_features, bias=False),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    """Low-rank input embedding for 8 Mimi codebooks."""

    def __init__(self, num_embeddings, embedding_dim, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(num_embeddings, rank),
                nn.GELU(),
                nn.Linear(rank, embedding_dim, bias=False),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, 8, T) audio token ids
        base_out = self.base(x)  # (B, 8, T, D)
        sum_out = base_out[:, 0, :, :]
        for i in range(self.num_layers):
            sum_out = sum_out + self.adapters[i](x[:, i, :])
        return sum_out / self.num_layers


# ═══════════════════════════════════════════════
# SenseVoice audio processor
# ═══════════════════════════════════════════════

class SenseVoiceAudioProcessor:
    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(self, wav, sampling_rate=16000, return_tensors="pt",
                 return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        attention_mask = (torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0)
        return SimpleNamespace(
            input_features=fbank,
            attention_mask=attention_mask,
        )


# ═══════════════════════════════════════════════
# Talker module
# ═══════════════════════════════════════════════

class TalkerModule(nn.Module):
    """Independent speech generation module (4 Versper blocks + low-rank codec I/O)."""

    def __init__(self, config):
        super().__init__()
        from types import SimpleNamespace as NS

        talker_cfg = NS(
            hidden_size=config.talker_hidden_size,
            use_moe=config.use_moe,
            dropout=config.dropout,
            flash_attn=config.flash_attn,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_act=config.hidden_act,
            intermediate_size=math.ceil(config.talker_hidden_size * math.pi / 64) * 64,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            norm_topk_prob=config.norm_topk_prob,
            router_aux_loss_coef=config.router_aux_loss_coef,
        )

        self.layers = nn.ModuleList([
            VersperBlock(l, talker_cfg)
            for l in range(config.num_talker_hidden_layers)
        ])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(
            config.talker_hidden_size, config.audio_vocab_size
        )
        self.embed_tokens = TalkerEmbedding(
            config.audio_vocab_size, config.talker_hidden_size
        )
        # Projection layers
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps),
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(
            config.spk_emb_size, config.talker_hidden_size, bias=False
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=talker_cfg.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


# ═══════════════════════════════════════════════
# Omni model (Thinker + Talker)
# ═══════════════════════════════════════════════

class VersperOmni(VersperForCausalLM):
    """Full omni model: text + speech + image input, text + streaming speech output."""

    config_class = OmniConfig

    def __init__(
        self,
        config: OmniConfig = None,
        audio_encoder_path="./model/SenseVoiceSmall",
        vision_model_path="./model/siglip2-base-p32-256-ve",
    ):
        config = config or OmniConfig()
        super().__init__(config)
        # Alias: self.thinker == self.model
        object.__setattr__(self, "thinker", self.model)
        object.__setattr__(self.model, "lm_head", self.lm_head)

        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(
            config.image_hidden_size, config.hidden_size
        )
        self.audio_pad_token = config.audio_pad_token
        self.audio_stop_token = config.audio_stop_token
        self.audio_spk_token = config.audio_spk_token

        # Load frozen encoders
        audio_encoder, audio_processor = self._load_sensevoice(audio_encoder_path)
        object.__setattr__(self, "audio_encoder", audio_encoder)
        object.__setattr__(self, "audio_processor", audio_processor)
        vision_encoder, vision_processor = self._load_vision(vision_model_path)
        object.__setattr__(self, "vision_encoder", vision_encoder)
        object.__setattr__(self, "vision_processor", vision_processor)

    # ── Encoder loading ──────────────────────────────────────

    @staticmethod
    def _load_sensevoice(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[VersperOmni] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel

            m = AutoModel(
                model=path,
                trust_remote_code=True,
                disable_update=True,
                device="cpu",
            )
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @staticmethod
    def _load_vision(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[VersperOmni] Vision path not found: {path}")
            return None, None
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    # ── Audio input encoding ─────────────────────────────────

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()):
            return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor(
                [valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device
            )
        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = []
        for i in range(emb.size(0)):
            seq_len = max(1, min(valid_lens[i].item(), emb.size(1)))
            projected = self.audio_proj(
                emb[i, :seq_len].unsqueeze(0).to(proj_dtype)
            ).squeeze(0)
            emb_list.append(projected)
        if batch_mask.all():
            return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def _inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb = h[b]
            seq = tokens[b].tolist()
            i = 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat(
                            (hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0
                        )
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    # ── Vision input encoding ────────────────────────────────

    @torch.compiler.disable
    def _get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, "keys"):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
            attn_mask = image_inputs.get("pixel_attention_mask")
            if attn_mask is not None and not attn_mask.any():
                return image_inputs["pixel_values"].new_zeros(
                    image_inputs["pixel_values"].size(0),
                    image_inputs["pixel_values"].size(1),
                    self.config.image_hidden_size,
                )
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def _encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None:
            return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any():
            return pixel_values.new_zeros(
                pixel_values.size(0),
                self.config.image_token_len,
                self.config.hidden_size,
            )
        with torch.no_grad():
            emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all():
            return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        result = emb.new_zeros(pixel_values.size(0), *emb.shape[1:])
        return result.scatter(0, idx, emb)

    @torch.compiler.disable
    def _inject_vision_features(self, tokens, h, vision_tensors, seqlen):
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

    # ── Forward pass ─────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        audio_inputs=None,
        audio_lens=None,
        pixel_values=None,
        **kwargs,
    ):
        if len(input_ids.shape) == 2:
            # Text only: (B, T)
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full(
                (batch_size, 8, seq_length),
                self.audio_pad_token,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            # Multi-stream: (B, 9, T) = 8 audio + 1 text
            batch_size, _, seq_length = input_ids.shape
            text_ids = input_ids[:, 8, :]
            audio_ids = input_ids[:, :8, :]

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        n_thinker = len(self.thinker.layers)
        n_talker = len(self.talker.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_talker))
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # RoPE buffers
        if self.thinker.freqs_cos[0, 0] == 0:
            c, s = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.thinker.freqs_cos = c.to(input_ids.device)
            self.thinker.freqs_sin = s.to(input_ids.device)
        if self.talker.freqs_cos[0, 0] == 0:
            c, s = precompute_freqs_cis(
                dim=self.talker.layers[0].self_attn.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.talker.freqs_cos = c.to(input_ids.device)
            self.talker.freqs_sin = s.to(input_ids.device)

        presents = []

        # ═══════ Thinker ═══════
        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(text_ids))
        position_embeddings = (
            self.thinker.freqs_cos[start_pos : start_pos + seq_length],
            self.thinker.freqs_sin[start_pos : start_pos + seq_length],
        )

        # Audio feature injection
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self._inject_audio_features(
                text_ids, hidden_states, audio_features, seq_length
            )

        # Vision feature injection
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, "keys"):
                img_emb = self._get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num = pixel_values.shape[:2]
                vision_tensors = torch.stack(
                    [self._encode_image_inputs(pixel_values[:, i]) for i in range(num)],
                    dim=1 if bs > 1 else 0,
                )
            hidden_states = self._inject_vision_features(
                tokens=text_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=seq_length,
            )

        bridge_states = hidden_states
        for i, (layer, pkv) in enumerate(
            zip(self.thinker.layers, past_key_values[:n_thinker])
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=pkv,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states
        h_thinker = self.thinker.norm(hidden_states)

        # ═══════ Talker ═══════
        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = kwargs.get("spk_emb", None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(
                spk_mask,
                self.talker.spk_proj(spk_emb).unsqueeze(1),
                talker_emb,
            )
        hidden_states = (
            self.talker.embed_proj(bridge_states) * self.talker.text_scale
            + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        )
        talker_pos_emb = (
            self.talker.freqs_cos[start_pos : start_pos + seq_length],
            self.talker.freqs_sin[start_pos : start_pos + seq_length],
        )
        for layer, pkv in zip(self.talker.layers, past_key_values[n_thinker:]):
            hidden_states, present = layer(
                hidden_states,
                talker_pos_emb,
                past_key_value=pkv,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        # ═══════ Output ═══════
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        aux_loss = sum(
            l.mlp.aux_loss
            for l in list(self.thinker.layers) + list(self.talker.layers)
            if isinstance(l.mlp, MOEFeedForward)
        )
        # DDP dummy gradients
        aux_loss = aux_loss + sum(p.sum() for p in self.audio_proj.parameters()) * 0
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
        aux_loss = (
            aux_loss
            + sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0
        )
        aux_loss = aux_loss + sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0

        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])

        out = MoeCausalLMOutputWithPast(
            aux_loss=aux_loss,
            logits=text_logits,
            past_key_values=presents,
        )
        out.audio_logits = audio_logits
        return out

    # ── Generation ──────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=1024,
        temperature=0.75,
        top_p=0.90,
        stream=False,
        rp=1.0,
        use_cache=True,
        return_audio_codes=False,
        **kwargs,
    ):
        if stream:
            return self._stream_generate(
                input_ids, eos_token_id, max_new_tokens,
                temperature, top_p, rp, use_cache, return_audio_codes, **kwargs,
            )
        tokens = list(
            self._stream_generate(
                input_ids, eos_token_id, max_new_tokens,
                temperature, top_p, rp, use_cache, return_audio_codes, **kwargs,
            )
        )
        return tokens[-1] if tokens else input_ids

    def _stream_generate(
        self, input_ids, eos_token_id, max_new_tokens,
        temperature, top_p, rp, use_cache, return_audio_codes, **kwargs,
    ):
        start_pos = input_ids.shape[1]
        past_kvs = None
        text_finished = False
        first_finished = True
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        audio_buffer = torch.full(
            (1, 8, start_pos),
            self.audio_pad_token,
            dtype=torch.long,
            device=input_ids.device,
        )
        spk_emb = kwargs.get("spk_emb", None)
        ref_codes = kwargs.get("ref_codes", None)
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0
        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            audio_buffer[:, :, fill_start:fill_end] = ref_codes[
                :, :, -(fill_end - fill_start):
            ]
        if spk_emb is not None and fill_start > 0:
            audio_buffer[:, :, fill_start - 1] = self.audio_spk_token

        think_end_step = None
        generated_tokens = [] if kwargs.get("open_thinking", False) else None

        while input_ids.shape[1] < start_pos + max_new_tokens:
            if past_kvs is None or not use_cache:
                out = self.forward(
                    torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1),
                    past_key_values=past_kvs,
                    use_cache=use_cache,
                    **kwargs,
                )
            else:
                out = self.forward(
                    torch.cat(
                        (audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1
                    ),
                    past_key_values=past_kvs,
                    use_cache=use_cache,
                    **kwargs,
                )
            past_kvs = out.past_key_values

            # Text token sampling
            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist()))
                score = logits[seen]
                logits[seen] = torch.where(
                    score > 0, score / rp, score * rp
                )
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float("Inf")
            text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            if text_finished:
                text_token = (
                    kwargs.get("enter_token_id", 201)
                    if first_finished
                    else kwargs.get("pad_token_id", 0)
                )
                first_finished = False

            step = input_ids.shape[1] - start_pos
            audio_step = step - 1  # delayed by 1 step

            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and generated_tokens[
                    -len(self.config.think_end_ids):
                ] == list(self.config.think_end_ids):
                    think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1

            # Audio code sampling per layer
            for i, al in enumerate(out.audio_logits):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    logits_i = al[0, -1, :].clone() / 0.2
                    # Repetition penalty on recent codes
                    for prev_code in audio_codes[i][-3:]:
                        score = logits_i[prev_code]
                        logits_i[prev_code] = torch.where(
                            score > 0, score / 1.05, score * 1.05
                        )
                    top_val, top_idx = logits_i.topk(50)
                    code = top_idx[
                        torch.multinomial(F.softmax(top_val, dim=-1), 1)
                    ].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            if text_finished and all(
                audio_stop_pos[i] is not None for i in range(8)
            ):
                break

            input_ids = torch.cat(
                (input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1
            )
            audio_buffer = torch.cat(
                (
                    audio_buffer,
                    torch.full(
                        (1, 8, 1),
                        self.audio_pad_token,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                ),
                dim=2,
            )
            for i in range(min(audio_step + 1, 8)):
                audio_buffer[0, i, -1] = audio_codes[i][-1]

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(
                    1
                    for i in range(8)
                    if audio_stop_pos[i] is None
                    or step - 7 + i < audio_stop_pos[i]
                )
                if active_layers >= 8:
                    audio_frame = frame

            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame


# ═══════════════════════════════════════════════
# Realtime VAD (zero-coupling utility)
# ═══════════════════════════════════════════════

class SileroVAD:
    def __init__(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self.h[:] = 0
        self.c[:] = 0

    def __call__(self, chunk, sr=16000):
        out, self.h, self.c = self.session.run(
            None,
            {
                "input": chunk.reshape(1, -1).astype(np.float32),
                "h": self.h,
                "c": self.c,
                "sr": np.array(sr, dtype="int64"),
            },
        )
        return float(out[0][0])


class RealtimeSession:
    def __init__(
        self,
        vad_path,
        sr=16000,
        threshold=0.8,
        min_speech_ms=128,
        min_silence_ms=800,
    ):
        self.vad = SileroVAD(vad_path)
        self.sr = sr
        self.threshold = threshold
        self.min_speech = int(sr * min_speech_ms / 1000)
        self.min_silence = int(sr * min_silence_ms / 1000)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.buffer = []
        self.ring = []
        self.speaking = False
        self.generating = False
        self.interrupt = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.tail_silence = 0

    def push_chunk(self, chunk, W=1024):
        for i in range(0, max(len(chunk), 1), W):
            w = chunk[i : i + W]
            if len(w) < W:
                w = np.pad(w, (0, W - len(w)))
            prob = self.vad(w, self.sr)
            if prob > self.threshold:
                self.silence_samples = 0
                self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.generating and self.speaking:
                    self.interrupt = True
                    return "interrupt"
            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1) :]
                    self.speaking = False
                    self.speech_samples = 0
                    self.silence_samples = 0
                    self.tail_silence = 0
                    return "speech_end"
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]
        return "listening"

    def get_audio(self):
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio
