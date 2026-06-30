"""
VLM (Vision-Language Model) SFT training.

Usage:
    torchrun --nproc_per_node 4 -m versper.trainer.sft_vlm --data_path ../dataset/sft_i2t.parquet
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse, time, warnings, torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from versper.config import VLMConfig
from versper.vlm import MiniMindVLM
from versper.dataset.vlm_dataset import VLMDataset, vlm_collate_fn
from versper.trainer.utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    log_model_params, save_checkpoint, load_checkpoint, SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, tb_writer=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps
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
            cur_lr = optimizer.param_groups[-1]["lr"]
            eta = spend / max(step - start_step, 1) * (iters - step) // 60
            Logger(f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {cur_loss:.4f}, lr: {cur_lr:.8f}, eta: {eta:.1f}min")
            if wandb:
                wandb.log({"loss": cur_loss, "lr": cur_lr})
            if tb_writer:
                global_step = epoch * iters + step
                tb_writer.add_scalar("loss", cur_loss, global_step)
                tb_writer.add_scalar("lr", cur_lr, global_step)
                tb_writer.add_scalar("eta_min", eta, global_step)
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_checkpoint(vlm_config, args.save_weight, model=model, optimizer=optimizer,
                            epoch=epoch, step=step, save_dir=args.save_dir, wandb=wandb)
            model.train()
        del input_ids, labels, pixel_values, res, loss
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VersperOmni VLM SFT")
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default="sft_vlm")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--vision_path", default="../model/siglip2-base-p32-256-ve")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=768, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", default="../dataset/sft_i2t.parquet")
    parser.add_argument("--from_weight", default="none")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--freeze_llm", default=1, type=int, choices=[0, 1, 2])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                           use_moe=bool(args.use_moe))
    ckp_data = load_checkpoint(vlm_config, args.save_weight, args.save_dir) if args.from_resume else None

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project="VersperOmni-VLM-SFT",
                   name=f"VLM-SFT-E{args.epochs}-B{args.batch_size}",
                   id=ckp_data.get("wandb_id") if ckp_data else None,
                   resume="must" if ckp_data and ckp_data.get("wandb_id") else None)

    tb_writer = None
    if args.use_tensorboard and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs", "sft_vlm"))
            Logger(f"TensorBoard logging to {args.save_dir}/runs/sft_vlm")
        except ModuleNotFoundError:
            Logger("tensorboard not installed; run: pip install tensorboard")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=args.vision_path)

    if args.from_weight != "none":
        wp = f"{args.save_dir}/{args.from_weight}_{args.hidden_size}{'_moe' if vlm_config.use_moe else ''}.pth"
        if os.path.exists(wp):
            model.load_state_dict(torch.load(wp, map_location=args.device), strict=False)

    # Freeze strategy
    for n, p in model.named_parameters():
        if "vision_proj" not in n:
            p.requires_grad = False
    if args.freeze_llm == 0:
        for n, p in model.named_parameters():
            if "vision_encoder" not in n:
                p.requires_grad = True
    elif args.freeze_llm == 1:
        last_idx = vlm_config.num_hidden_layers - 1
        for n, p in model.model.named_parameters():
            if "layers.0." in n or f"layers.{last_idx}." in n:
                p.requires_grad = True
    # freeze_llm == 2: only vision_proj trains (default)

    log_model_params(model)
    model = model.to(args.device)

    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=model.processor,
                          image_special_token=vlm_config.image_special_token,
                          image_token_len=vlm_config.image_token_len,
                          max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)

    if args.use_compile:
        model = torch.compile(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=vlm_collate_fn)
        if skip > 0:
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, tb_writer)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, tb_writer)

    if dist.is_initialized():
        dist.destroy_process_group()
    if tb_writer:
        tb_writer.close()
