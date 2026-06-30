# 有监督微调 — SFT 数据与训练

> 对应脚本：`src/versper/trainer/sft_vlm.py`（VLM SFT）、`src/versper/trainer/sft_omni.py`（Omni SFT）

## 概述

有监督微调（Supervised Fine-Tuning, SFT）在预训练模型基础上，使用高质量的对话/指令数据对模型进行微调，使其具备对话交互、指令遵循和工具调用等能力。根据训练目标不同，SFT 分为：

- **LM SFT**：纯文本对话微调，使用 `pretrain.py` + `SFTDataset`
- **VLM SFT**：视觉-语言多模态微调，使用 `sft_vlm.py`
- **Omni SFT**：全模态（语音+视觉+文本）微调，使用 `sft_omni.py`

## 数据格式

SFT 数据集为 `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`（纯文本）和 `sft_i2t.parquet`（图文对话），均采用统一的对话格式。

### 纯文本对话格式

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。请查看天气预报应用。"}
    ]
}
```

### 带工具调用（Tool Calling）格式

```jsonl
{
    "conversations": [
        {"role": "system", "content": "你是一个助手，可以使用工具帮助用户。", "tools": "[{\"name\":\"translate_text\",\"description\":\"翻译文本\",\"parameters\":{...}}]"},
        {"role": "user", "content": "把'你好世界'翻译成英文"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

## 训练脚本

### LM SFT（文本对话微调）

通过 `pretrain.py` 的 `SFTDataset` 实现，使用较小的学习率进行微调：

```bash
python -m versper.trainer.pretrain \
    --data_path ../dataset/sft_t2t.jsonl \
    --from_weight pretrain \
    --learning_rate 2e-5 \
    --epochs 3
```

### VLM SFT（视觉-语言微调）

```bash
torchrun --nproc_per_node 4 -m versper.trainer.sft_vlm \
    --data_path ../dataset/sft_i2t.parquet \
    --freeze_llm 1 \
    --learning_rate 5e-6 \
    --epochs 3
```

### Omni SFT（全模态微调）

```bash
torchrun --nproc_per_node 4 -m versper.trainer.sft_omni \
    --data_path ../dataset/sft_omni.parquet \
    --freeze_backbone last1 \
    --learning_rate 5e-4 \
    --mode all
```

## `sft_vlm.py` 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_dir` | `../out` | 模型保存根目录 |
| `--save_weight` | `sft_vlm` | 权重保存名称 |
| `--tokenizer_path` | `../model` | tokenizer 所在目录 |
| `--vision_path` | `../model/siglip2-base-p32-256-ve` | 视觉编码器路径（SigLIP2） |
| `--epochs` | `2` | 训练轮数 |
| `--batch_size` | `4` | 每个 GPU 的 batch size（VLM 显存占用较大） |
| `--learning_rate` | `5e-6` | 峰值学习率 |
| `--device` | `cuda:0` | 训练设备 |
| `--dtype` | `bfloat16` | 训练精度 |
| `--num_workers` | `2` | DataLoader 工作进程数 |
| `--accumulation_steps` | `1` | 梯度累积步数 |
| `--grad_clip` | `1.0` | 梯度裁剪阈值 |
| `--log_interval` | `100` | 日志打印间隔 |
| `--save_interval` | `1000` | checkpoint 保存间隔 |
| `--hidden_size` | `768` | Transformer 隐层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--max_seq_len` | `768` | 最大序列长度 |
| `--use_moe` | `0` | 是否使用 MoE，可选 0 / 1 |
| `--data_path` | `../dataset/sft_i2t.parquet` | 训练数据路径 |
| `--from_weight` | `none` | 加载预训练权重 |
| `--from_resume` | `0` | 是否从 checkpoint 恢复 |
| `--freeze_llm` | `1` | LLM 冻结模式，可选 0 / 1 / 2 |
| `--use_wandb` | — | 启用 W&B 日志（action flag） |
| `--use_compile` | `0` | 是否使用 `torch.compile`，可选 0 / 1 |

## `sft_omni.py` 特有参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--audio_encoder_dir` | `../model/SenseVoiceSmall` | 语音编码器路径（SenseVoice） |
| `--vision_dir` | `../model/siglip2-base-p32-256-ve` | 视觉编码器路径 |
| `--freeze_backbone` | `none` | 冻结骨干网络模式：`none`（全部可训练）、`all`（全部冻结）、`last1`（仅冻结除最后一层外的 LLM） |
| `--mode` | `all` | 训练模块选择：`all`（全体）、`audio_proj`（仅语音投影层）、`vision_proj`（仅视觉投影层） |

## `freeze_llm` 模式详解（VLM SFT）

| freeze_llm 值 | 行为 | 适用场景 |
|:---:|---|---|
| `0` | 全部解冻，全参数微调 | 数据充足、需深度对齐时使用 |
| `1`（默认） | 仅训练首尾两层 LLM + vision_proj 投影层 | 平衡微调效果与训练效率 |
| `2` | 仅训练 vision_proj 投影层，LLM 完全冻结 | 初始视觉适配阶段，防止 LLM 灾难性遗忘 |

## 训练技巧与建议

### 学习率选择

- **全参数微调（freeze_llm=0）**：建议使用较小学习率 `1e-5` ~ `5e-6`，避免破坏预训练权重。
- **部分微调（freeze_llm=1/2）**：可使用稍高学习率 `5e-6` ~ `1e-5`。
- 视觉投影层由于是随机初始化，在训练初期可能需要相对较高的学习率。

### 批量大小与显存

- VLM 训练由于需要同时加载视觉编码器和 LLM，显存占用较大。默认 `batch_size=4`。
- 可通过 `--accumulation_steps` 弥补小 batch 带来的梯度噪声。
- Omni 训练在 VLM 基础上增加了语音编码器，建议先使用 `--mode audio_proj` 单独训练语音投影层。

### 数据配比

- 纯文本对话数据与图文对话数据可按比例混合输入。
- 建议保留 5%~10% 的纯文本数据以维持对话能力，防止多模态训练导致的语言能力退化。
- 工具调用数据已合并至主分支 SFT 数据中，训练时自动参与 loss 计算。
