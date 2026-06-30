"""
Omni Model SFT training (T2A / A2A / I2T modes).

Usage:
    # Full model T2A training
    torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \\
        --data_path ../dataset/sft_t2a.parquet --mode all --lr 5e-6

    # Audio projector only
    torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \\
        --data_path ../dataset/sft_a2a.parquet --mode audio_proj --lr 5e-4

    # Vision projector only
    torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \\
        --data_path ../dataset/sft_i2t.parquet --mode vision_proj --lr 5e-5
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse, time, warnings, torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from versper.config import OmniConfig
from versper.omni import MiniMindOmni
from versper.dataset.omni_dataset import OmniDataset, omni_collate_fn
from versper.trainer.utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    log_model_params, save_checkpoint, load_checkpoint, SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tb_writer=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels, audio_labels, audio_inputs, audio_lens,
               pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens,
                        pixel_values=pixel_values, spk_emb=spk_emb)
            loss_fct = nn.CrossEntropyLoss(reduction="none")

            # Text loss
            text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
            text_mask = (labels.view(-1) != -100).float()
            text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)

            # Audio loss (8 codebook layers)
            audio_loss = res.audio_logits[0].sum() * 0
            for i, al in enumerate(res.audio_logits):
                al_flat = al.view(-1, al.size(-1))
                target_flat = audio_labels[:, i, :].reshape(-1)
                layer_loss = loss_fct(al_flat, target_flat)
                valid_mask = (target_flat != -100).float()
                stop_mask = (target_flat == 2050).float()
                weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)
                msum = valid_mask.sum()
                if msum > 0:
                    audio_loss = audio_loss + weighted_loss.sum() / (msum + 1e-9)
            audio_loss = audio_loss / 8

            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            tl = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            al = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            cur_lr = optimizer.param_groups[-1]["lr"]
            eta = spend / max(step - start_step, 1) * (iters - step) // 60
            Logger(f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {cur_loss:.4f}, "
                   f"text: {tl:.4f}, audio: {al:.4f}, lr: {cur_lr:.8f}, eta: {eta:.1f}min")
            if wandb:
                wandb.log({"loss": cur_loss, "text_loss": tl, "audio_loss": al, "lr": cur_lr})
            if tb_writer:
                global_step = epoch * iters + step
                tb_writer.add_scalar("loss", cur_loss, global_step)
                tb_writer.add_scalar("text_loss", tl, global_step)
                tb_writer.add_scalar("audio_loss", al, global_step)
                tb_writer.add_scalar("lr", cur_lr, global_step)
                tb_writer.add_scalar("eta_min", eta, global_step)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_checkpoint(omni_config, args.save_weight, model=model, optimizer=optimizer,
                            epoch=epoch, step=step, save_dir=args.save_dir, wandb=wandb)
            model.train()

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VersperOmni Omni SFT")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default="sft_omni")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--audio_encoder_dir", default="../model/SenseVoiceSmall")
    parser.add_argument("--vision_dir", default="../model/siglip2-base-p32-256-ve")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", default="../dataset/sft_t2a.parquet")
    parser.add_argument("--from_weight", default="none")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--freeze_backbone", default="none", choices=["none", "all", "last1"])
    parser.add_argument("--mode", default="all", choices=["all", "audio_proj", "vision_proj"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    omni_config = OmniConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             use_moe=bool(args.use_moe))
    ckp_data = load_checkpoint(omni_config, args.save_weight, args.save_dir) if args.from_resume else None

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project="VersperOmni-Omni-SFT",
                   name=f"Omni-{args.mode}-E{args.epochs}-B{args.batch_size}",
                   id=ckp_data.get("wandb_id") if ckp_data else None,
                   resume="must" if ckp_data and ckp_data.get("wandb_id") else None)

    tb_writer = None
    if args.use_tensorboard and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs", "sft_omni"))
            Logger(f"TensorBoard logging to {args.save_dir}/runs/sft_omni")
        except ModuleNotFoundError:
            Logger("tensorboard not installed; run: pip install tensorboard")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindOmni(omni_config, audio_encoder_path=args.audio_encoder_dir,
                         vision_model_path=args.vision_dir)

    if args.from_weight != "none":
        wp = f"{args.save_dir}/{args.from_weight}_{args.hidden_size}{'_moe' if omni_config.use_moe else ''}.pth"
        if os.path.exists(wp):
            weights = torch.load(wp, map_location=args.device)
            shape_mismatch = {k for k, v in weights.items()
                              if k in dict(model.named_parameters()) and
                              v.shape != dict(model.named_parameters())[k].shape}
            if shape_mismatch:
                Logger(f"Skipping mismatched keys: {shape_mismatch}")
                weights = {k: v for k, v in weights.items() if k not in shape_mismatch}
            model.load_state_dict(weights, strict=False)
            Logger(f"Loaded weights: {wp}")
            # Initialize Talker from Thinker if no Talker weights
            if args.from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                has_talker = any(k.startswith("talker.layers.") for k in weights)
                if not has_talker and omni_config.num_talker_hidden_layers > 0:
                    n_talker = omni_config.num_talker_hidden_layers
                    n_thinker = len(model.thinker.layers)
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(
                            model.thinker.layers[src].state_dict()
                        )
                    Logger(f"Talker initialized from thinker layers[{n_thinker-n_talker}:{n_thinker}]")

    # Freeze backbone
    if args.freeze_backbone == "all":
        for p in model.model.parameters():
            p.requires_grad = False
    elif args.freeze_backbone == "last1":
        for p in model.model.parameters():
            p.requires_grad = False
        if hasattr(model.model, "layers") and len(model.model.layers) > 0:
            for p in model.model.layers[-1].parameters():
                p.requires_grad = True

    # Mode-specific freeze
    if args.mode == "audio_proj":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.audio_proj.parameters():
            p.requires_grad = True
    elif args.mode == "vision_proj":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.vision_proj.parameters():
            p.requires_grad = True

    log_model_params(model)
    model = model.to(args.device)

    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)
    if model.vision_encoder is not None:
        model.vision_encoder.to(args.device)

    if args.use_compile:
        model = torch.compile(model)

    train_ds = OmniDataset(args.data_path, tokenizer, audio_processor=model.audio_processor,
                           vision_processor=model.vision_processor, max_length=args.max_seq_len,
                           image_token_len=omni_config.image_token_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, tb_writer)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, tb_writer)

    if dist.is_initialized():
        dist.destroy_process_group()
    if tb_writer:
        tb_writer.close()
