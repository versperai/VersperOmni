# 预训练 — 数据、脚本与训练流程

> 对应脚本：`src/versper/trainer/pretrain.py`

## 概述

预训练阶段采用**自回归语言建模**（next-token prediction）目标，在大规模纯文本语料上训练 Transformer 模型。模型根据上文预测下一个 token，通过交叉熵损失函数优化。该阶段为模型提供广泛的语言知识、事实基础和中英双语能力。

## 数据格式

预训练数据集为 `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`，采用 JSONL 格式，每行为一个 JSON 对象：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

每条数据仅包含 `text` 字段，训练时对该文本序列进行完整的 next-token prediction 损失计算。

- `pretrain_t2t_mini.jsonl`：适用于快速验证与复现的迷你版本
- `pretrain_t2t.jsonl`：完整预训练主分支使用的全量版本

## 训练脚本

### 单卡训练

```bash
python -m versper.trainer.pretrain --data_path ../dataset/pretrain_t2t.jsonl
```

### 多卡 DDP 训练

```bash
torchrun --nproc_per_node 4 -m versper.trainer.pretrain \
    --data_path ../dataset/pretrain_t2t.jsonl \
    --batch_size 32 \
    --epochs 2 \
    --learning_rate 5e-4 \
    --use_wandb
```

## 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_dir` | `../out` | 模型保存根目录 |
| `--save_weight` | `pretrain` | 权重保存名称（子目录名） |
| `--tokenizer_path` | `../model` | tokenizer（tokenizer.json / tokenizer_config.json）所在目录 |
| `--epochs` | `2` | 训练轮数 |
| `--batch_size` | `32` | 每个 GPU 的 batch size |
| `--learning_rate` | `5e-4` | 峰值学习率 |
| `--device` | `cuda:0` | 训练设备（自动检测 CUDA） |
| `--dtype` | `bfloat16` | 训练精度（bfloat16 / float32 等） |
| `--num_workers` | `8` | DataLoader 工作进程数 |
| `--accumulation_steps` | `8` | 梯度累积步数 |
| `--grad_clip` | `1.0` | 梯度裁剪阈值 |
| `--log_interval` | `100` | 日志打印间隔（步数） |
| `--save_interval` | `1000` | checkpoint 保存间隔（步数） |
| `--hidden_size` | `768` | Transformer 隐层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--max_seq_len` | `340` | 最大序列长度（超出部分截断） |
| `--use_moe` | `0` | 是否使用 MoE（Mixture of Experts）架构，可选 0 / 1 |
| `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` | 训练数据路径 |
| `--from_weight` | `none` | 加载预训练权重，传入 save_weight 名称 |
| `--from_resume` | `0` | 是否从最近 checkpoint 恢复训练，可选 0 / 1 |
| `--use_wandb` | — | 启用 Weights & Biases 日志记录（action flag） |
| `--wandb_project` | `VersperOmni-Pretrain` | W&B 项目名称 |
| `--use_compile` | `0` | 是否使用 `torch.compile` 加速，可选 0 / 1 |

## 训练技巧与建议

### Batch Size 与梯度累积

- 单卡 batch size 推荐设置为 `32`（默认值）。
- 通过 `--accumulation_steps` 实现**有效大 batch**：`有效 batch size = batch_size × accumulation_steps × GPU 数`。
- 例如 4 卡 + accumulation_steps=8，有效 batch size 为 `32 × 8 × 4 = 1024`。

### 混合精度训练

默认使用 `bfloat16` 混合精度，在支持 bf16 的 GPU（如 A100、H100、RTX 4090）上可获得更好的训练稳定性和吞吐量。如果硬件不支持 bf16，可切换为 `float32`。

### 监控与损失曲线

- 训练初期 loss 通常在 **6~7** 左右，随训练步数增加应平滑下降。
- 若 loss 出现剧烈震荡或上升，可检查学习率是否过大、数据是否存在异常。
- 使用 `--use_wandb` 可实时监控 loss 曲线、学习率变化和梯度范数。

## 断点续训与权重加载

### 从 checkpoint 恢复

```bash
python -m versper.trainer.pretrain --data_path ../dataset/pretrain_t2t.jsonl --from_resume 1
```

`--from_resume 1` 会自动加载 `save_dir/save_weight` 目录下最新的 checkpoint（包含优化器状态），从断点处继续训练。

### 加载预训练权重

```bash
python -m versper.trainer.pretrain --data_path ../dataset/pretrain_t2t.jsonl --from_weight pretrain
```

`--from_weight pretrain` 会从 `save_dir/pretrain` 目录加载模型权重，适用于继续训练或作为下游任务的初始化。注意 `from_weight` 和 `from_resume` 的区别：前者仅加载模型参数，后者同时恢复优化器状态。
