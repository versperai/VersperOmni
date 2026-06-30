# 训练脚本指南 -- pretrain / sft_vlm / sft_omni

## 概述

三个训练脚本均位于 `src/versper/trainer/` 目录下，共享 `trainer/utils.py` 中的公共工具函数。所有脚本都是独立入口点（`if __name__ == "__main__"`），使用 `argparse` 解析参数。

| 脚本 | 文件 | 模型 | 配置类 |
|------|------|------|--------|
| `pretrain.py` | `trainer/pretrain.py` | LM 预训练 | `MiniMindConfig` |
| `sft_vlm.py` | `trainer/sft_vlm.py` | VLM 微调 | `VLMConfig` |
| `sft_omni.py` | `trainer/sft_omni.py` | Omni 微调 | `OmniConfig` |

---

## 共享工具函数 (`trainer/utils.py`)

### 分布式初始化

```python
def init_distributed_mode():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0  # 单进程模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 学习率调度

```python
def get_lr(current_step, total_steps, lr):
    """Cosine 衰减: lr → 0.1*lr"""
    return lr * (0.1 + 0.45 * (1 + cos(π * current_step / total_steps)))
```

Cosine 衰减从 `lr` 开始，到 `0.1 * lr` 结束。

### 参数量统计

```python
def log_model_params(model, ignore_patterns=None):
    # ignore_patterns 默认忽略 audio_encoder, vision_encoder
    # 计算总参数量和激活参数量 (MoE 场景)
```

### 保存/加载 Checkpoint

```python
def save_checkpoint(config, weight_name, model, optimizer, epoch, step, ...):
    # 保存两个文件:
    #   1. {weight_name}_{hidden_size}{_moe}.pth         -- 权重文件
    #   2. {weight_name}_{hidden_size}{_moe}_resume.pth   -- 恢复文件 (含优化器状态)
    # 自动移除 vision_encoder. 和 audio_encoder. 的冻结参数

def load_checkpoint(config, weight_name, save_dir):
    # 加载恢复文件，检测 GPU 数量变化并调整 step
```

### SkipBatchSampler

用于断点续训时跳过已处理的 batch：

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        # 跳过前 skip_batches 个 batch
```

---

## Pretrain Script (`pretrain.py`)

### 用法

```bash
# 单卡
python -m versper.trainer.pretrain \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --tokenizer_path ../model

# 多卡 DDP (4 GPU)
torchrun --nproc_per_node 4 -m versper.trainer.pretrain \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --tokenizer_path ../model \
    --batch_size 32 \
    --epochs 2

# 从预训练权重加载
torchrun --nproc_per_node 4 -m versper.trainer.pretrain \
    --from_weight pretrain

# 从 checkpoint 恢复
torchrun --nproc_per_node 4 -m versper.trainer.pretrain \
    --from_resume 1
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 32 | 每 GPU batch size |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--epochs` | 2 | 训练轮数 |
| `--max_seq_len` | 340 | 序列最大长度 |
| `--accumulation_steps` | 8 | 梯度累积步数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--use_moe` | 0 | 是否使用 MoE |
| `--use_compile` | 0 | 是否使用 torch.compile |
| `--from_weight` | "none" | 加载预训练权重路径 |
| `--from_resume` | 0 | 从 checkpoint 恢复 |
| `--save_interval` | 1000 | 保存间隔 |
| `--hidden_size` | 768 | 模型维度 |
| `--num_hidden_layers` | 8 | Transformer 层数 |

### 训练流程

```python
# 1. 初始化分布式
local_rank = init_distributed_mode()

# 2. 创建配置和模型
lm_config = MiniMindConfig(hidden_size=args.hidden_size, ...)
model = MiniMindForCausalLM(lm_config)

# 3. 加载权重 (可选)
if args.from_weight != "none":
    model.load_state_dict(torch.load(weight_path), strict=False)

# 4. 混合精度 + 优化器
autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

# 5. DDP 封装
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])

# 6. 训练循环
for epoch in range(epochs):
    for step, (input_ids, labels) in enumerate(loader):
        loss = model(input_ids, labels=labels).loss
        scaler.scale(loss).backward()
        # 梯度累积 + 裁剪 + 更新
```

### 恢复训练

当 `--from_resume 1` 时：
1. 加载 `_resume.pth` 中的 `model`, `optimizer`, `scaler`, `epoch`, `step`
2. 使用 `SkipBatchSampler` 跳过已处理的 batch
3. SwanLab (wandb) 也可恢复运行 ID

---

## VLM SFT Script (`sft_vlm.py`)

### 用法

```bash
# VLM 微调 (默认 freeze_llm=1, 仅微调首尾层 + projection)
torchrun --nproc_per_node 4 -m versper.trainer.sft_vlm \
    --data_path ../dataset/sft_i2t.parquet \
    --vision_path ../model/siglip2-base-p32-256-ve

# 全参数微调
torchrun --nproc_per_node 4 -m versper.trainer.sft_vlm \
    --data_path ../dataset/sft_i2t.parquet \
    --freeze_llm 0

# 仅微调 projection
torchrun --nproc_per_node 4 -m versper.trainer.sft_vlm \
    --data_path ../dataset/sft_i2t.parquet \
    --freeze_llm 2
```

### Freeze LLM 模式

```python
# freeze_llm=0: 全部 unfreeze (vision_encoder 除外)
for n, p in model.named_parameters():
    if "vision_encoder" not in n:
        p.requires_grad = True

# freeze_llm=1 (默认): 仅首尾层 + vision_proj
for n, p in model.model.named_parameters():
    if "layers.0." in n or f"layers.{last_idx}." in n:
        p.requires_grad = True

# freeze_llm=2: 仅 vision_proj
# (默认冻结所有，但 vision_proj 在初始时已设为 requires_grad=True)
```

| `freeze_llm` | 可训练参数 |
|:---:|------|
| 0 | 全部（除 vision_encoder）|
| 1 | vision_proj + LLM 首层 + LLM 末层 |
| 2 | 仅 vision_proj |

### 关键差异

| 项目 | pretrain.py | sft_vlm.py |
|------|-------------|-------------|
| 学习率 | 5e-4 | 5e-6 (小很多) |
| Batch size | 32 | 4 |
| 最大长度 | 340 | 768 |
| 梯度累积 | 8 | 1 |
| DDP ignore | 无 | `freqs_cos`, `freqs_sin` |

```python
# VLM 需要忽略 RoPE buffer 的 DDP 同步
model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
```

---

## Omni SFT Script (`sft_omni.py`)

### 用法

```bash
# 全模型训练 (Thinker+Talker)
torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \
    --data_path ../dataset/sft_t2a.parquet \
    --mode all \
    --learning_rate 5e-6

# 仅训练 Audio Projector
torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \
    --data_path ../dataset/sft_a2a.parquet \
    --mode audio_proj \
    --learning_rate 5e-4

# 仅训练 Vision Projector
torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \
    --data_path ../dataset/sft_i2t.parquet \
    --mode vision_proj \
    --learning_rate 5e-5
```

### 训练模式

```python
if args.mode == "audio_proj":
    # 冻结全部，仅训练 audio_proj
    for p in model.audio_proj.parameters():
        p.requires_grad = True
elif args.mode == "vision_proj":
    # 冻结全部，仅训练 vision_proj
    for p in model.vision_proj.parameters():
        p.requires_grad = True
# mode == "all": 所有参数可训练 (受 freeze_backbone 控制)
```

### Backbone Freeze 模式

```python
if args.freeze_backbone == "all":
    # 冻结整个 thinker backbone
    for p in model.model.parameters():
        p.requires_grad = False
elif args.freeze_backbone == "last1":
    # 冻结 thinker backbone，仅保留最后一层
    for p in model.model.layers[-1].parameters():
        p.requires_grad = True
```

### 损失函数

Omni 训练使用组合损失：

```python
# 文本损失 (CrossEntropy, ignore -100)
text_loss = F.cross_entropy(logits, labels, ignore_index=-100)

# 音频损失 (8 codebook layers, 加权)
for i in range(8):
    layer_loss = F.cross_entropy(audio_logits[i], audio_labels[:, i, :])
    # stop token 权重 ×10 (1 + 9)
    weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)

loss = (text_loss + audio_loss + aux_loss) / accumulation_steps
```

音频损失对 `audio_stop_token=2050` 的权重增加 10 倍，鼓励模型准确预测停止位置。

### 权重初始化与 Talker 拷贝

从 LM 权重初始化 Omni 时自动初始化 Talker：

```python
if args.from_weight != "none":
    weights = torch.load(weight_path)
    model.load_state_dict(weights, strict=False)
    
    # 自动拷贝 Thinker 最后 N 层到 Talker
    if (not has_talker_weights and 
        talker_hidden_size == hidden_size and 
        num_talker_layers > 0):
        for i in range(num_talker_layers):
            src = num_thinker_layers - num_talker_layers + i
            model.talker.layers[i].load_state_dict(
                model.thinker.layers[src].state_dict()
            )
```

---

## DDP 训练通用流程

所有三个脚本的 DDP 流程一致：

```bash
torchrun --nproc_per_node {GPU数量} -m versper.trainer.{script} \
    [--args...]
```

```python
# 1. 解析环境变量 RANK, LOCAL_RANK, WORLD_SIZE
local_rank = init_distributed_mode()

# 2. 设置设备
args.device = f"cuda:{local_rank}"

# 3. 设置种子 (不同 rank 偏移不同 seed)
setup_seed(42 + dist.get_rank())

# 4. DistributedSampler
train_sampler = DistributedSampler(train_ds)

# 5. DDP 封装
model = DistributedDataParallel(model, device_ids=[local_rank])

# 6. Epoch 循环前设置 sampler epoch
train_sampler.set_epoch(epoch)

# 7. 结束
dist.destroy_process_group()
```
