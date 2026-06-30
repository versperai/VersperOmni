"""
Shared training utilities for all VersperOmni variants.
Consolidated from minimind/minimind-v/minimind-o trainer_utils.
"""
import os
import random
import math
import torch
import torch.distributed as dist
import numpy as np
import datasets  # noqa: F401  # Windows pyarrow/torch DLL workaround
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════
# Distributed / Seed / Logging
# ═══════════════════════════════════════════════

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def init_distributed_mode():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0  # single process
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════
# LR schedule
# ═══════════════════════════════════════════════

def get_lr(current_step, total_steps, lr):
    """Cosine decay from lr to 0.1*lr."""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ═══════════════════════════════════════════════
# Parameter counting
# ═══════════════════════════════════════════════

def log_model_params(model, ignore_patterns=None):
    """Log total and active parameter counts."""
    if ignore_patterns is None:
        ignore_patterns = ["audio_encoder", "vision_encoder"]

    def should_count(n):
        return not any(p in n for p in ignore_patterns)

    total = (
        sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    )
    cfg = model.config
    n_routed = getattr(cfg, "n_routed_experts", getattr(cfg, "num_experts", 0))
    n_active = getattr(cfg, "num_experts_per_tok", 0)
    n_shared = getattr(cfg, "n_shared_experts", 0)
    expert = (
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if "mlp.experts.0." in n and should_count(n)
        )
        / 1e6
    )
    shared_expert = (
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if "mlp.shared_experts.0." in n and should_count(n)
        )
        / 1e6
    )
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


# ═══════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════

def save_checkpoint(
    config,
    weight_name,
    model,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="../checkpoints",
    state_dict=None,
    **kwargs,
):
    """Save model weight + resume checkpoint. Compatible with LM/VLM/Omni."""
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = "_moe" if config.use_moe else ""
    ckp_path = f"{save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}.pth"
    resume_path = f"{save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}_resume.pth"

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)

    if state_dict is None:
        state_dict = raw_model.state_dict()
    # Remove frozen encoder params (vision_encoder, audio_encoder)
    clean = {
        k: v.half().cpu()
        for k, v in state_dict.items()
        if not k.startswith("vision_encoder.") and not k.startswith("audio_encoder.")
    }
    ckp_tmp = ckp_path + ".tmp"
    torch.save(clean, ckp_tmp)
    os.replace(ckp_tmp, ckp_path)

    wandb_id = None
    if wandb:
        if hasattr(wandb, "get_run"):
            run = wandb.get_run()
            wandb_id = getattr(run, "id", None) if run else None
        else:
            wandb_id = getattr(wandb, "id", None)

    resume_data = {
        "model": clean,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": wandb_id,
    }
    for key, value in kwargs.items():
        if value is not None:
            if hasattr(value, "state_dict"):
                rv = value.module if isinstance(value, DistributedDataParallel) else value
                rv = getattr(rv, "_orig_mod", rv)
                resume_data[key] = rv.state_dict()
            else:
                resume_data[key] = value

    rtmp = resume_path + ".tmp"
    torch.save(resume_data, rtmp)
    os.replace(rtmp, resume_path)
    del clean
    torch.cuda.empty_cache()


def load_checkpoint(config, weight_name, save_dir="../checkpoints"):
    """Load resume checkpoint. Returns None if not found."""
    moe_suffix = "_moe" if config.use_moe else ""
    resume_path = f"{save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}_resume.pth"
    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path, map_location="cpu")
        saved_ws = ckp_data.get("world_size", 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
            Logger(
                f"GPU count change ({saved_ws}→{current_ws}), "
                f"step adjusted to {ckp_data['step']}"
            )
        return ckp_data
    return None


def load_weight_path(config, weight_name, save_dir="../out"):
    moe_suffix = "_moe" if config.use_moe else ""
    return f"{save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}.pth"


# ═══════════════════════════════════════════════
# Batch Sampler
# ═══════════════════════════════════════════════

class SkipBatchSampler(Sampler):
    """Skip N batches from start (for resume training)."""

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)
