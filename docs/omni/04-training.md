# 训练流程详解

> 逐步递进的多阶段训练策略

## 1. 训练模式

训练脚本 `train_sft_omni.py` 支持三种模式，通过命令行开关控制：

| 模式 | 更新参数 | 冻结参数 | 用途 |
|------|---------|---------|------|
| **all** | Thinker + Talker + 所有投影器 | 无 | 全模型联合训练 |
| **audio_proj** | 仅音频投影器 | Thinker + Talker + 视觉投影器 | 音频编码器适配 |
| **vision_proj** | 仅视觉投影器 | Thinker + Talker + 音频投影器 | 视觉编码器适配 |

> 注意：SenseVoice 和 SigLIP2 编码器始终冻结，Mimi codec 也始终冻结。

## 2. 训练阶段

完整的训练按以下顺序分阶段进行：

### Stage 1: 全模型 T2A（文本→语音）
| 配置 | 值 |
|------|-----|
| 数据 | sft_t2a（1,248,923 条） |
| 模式 | **all**（全模型更新） |
| 学习率 | 5 × 10⁻⁶ |
| Epochs | 1 |
| 用时 | ~45 分钟 |
| 目的 | 让模型学会将文本指令映射为语音输出 |

### Stage 2: 音频投影器 A2A（语音→语音，投影器适配）
| 配置 | 值 |
|------|-----|
| 数据 | sft_a2a（414,024 条） |
| 模式 | **audio_proj**（仅音频投影器） |
| 学习率 | 5 × 10⁻⁴ |
| Epochs | 1 |
| 用时 | ~25 分钟 |
| 目的 | 适配音频编码器输出到 Thinker embedding space |

### Stage 3: 全模型 A2A（语音→语音，联合训练）
| 配置 | 值 |
|------|-----|
| 数据 | sft_a2a |
| 模式 | **all**（全模型更新） |
| 学习率 | 5 × 10⁻⁵ |
| Epochs | 3 |
| 用时 | ~75 分钟 |
| 目的 | 充分学习语音输入/输出闭环 |

### Stage 4: 全模型 I2T（图像→文本）
| 配置 | 值 |
|------|-----|
| 数据 | sft_i2t（~100K 条） |
| 模式 | **all**（全模型更新） |
| 学习率 | 5 × 10⁻⁶ |
| Context | 768 tokens |
| Epochs | 1 |
| 用时 | ~45 分钟 |
| 目的 | 为模型添加视觉理解能力 |

### Stage 5: 视觉投影器 I2T（图像→文本，投影器适配）
| 配置 | 值 |
|------|-----|
| 数据 | sft_i2t |
| 模式 | **vision_proj**（仅视觉投影器） |
| 学习率 | 5 × 10⁻⁵ |
| Epochs | 1 |
| 用时 | ~45 分钟 |
| 目的 | 精调视觉投影器，不干扰已学到的语言/语音能力 |

### 总耗时

**Dense 或 MoE 单变体完整训练 ≈ 4 小时**（在 4× RTX 3090 上）

## 3. 训练超参数

| 超参数 | 值 |
|--------|-----|
| GPU | 4 × NVIDIA RTX 3090 (24GB) |
| 分布式 | PyTorch DDP (`torchrun --nproc_per_node 4`) |
| 精度 | bf16 mixed precision |
| 优化器 | AdamW |
| Batch size (per GPU) | 32 |
| 梯度累积 | 无 |
| 梯度裁剪 | 1.0 |
| 学习率调度 | 固定学习率（各阶段不同） |

## 4. Dense vs MoE 变体

| 方面 | Dense | MoE |
|------|-------|-----|
| 总参数量 | ~115.29M | ~317.05M |
| 活跃参数量 | ~115.29M | ~115.33M |
| Thinker | 8 层 Dense FFN | MoE FFN（多个 expert） |
| Talker | 4 层 Dense FFN | MoE FFN（多个 expert） |
| 训练曲线 | 平滑下降 | 与 Dense 接近 |
| 最佳 CER | 0.0897 | 0.0900 |
| 说话人相似度 | 0.5995 | 0.5937 |

MoE 本质上是**容量分配实验**而非最终的 expert 布局。其曲线应被解读为在相似活跃参数下容量分配方式的比较。

## 5. 模块配置详情

| 模块 | 具体结构 | 关键配置 | 状态 / 参数量 |
|------|---------|---------|-------------|
| Thinker | MiniMind Transformer | 8层, hidden 768, 8/4 KV heads, vocab 6400 | trainable, 63.91M / 198.42M |
| Talker | 4 × MiniMind blocks | hidden 768, audio vocab 2112, 8 heads, rank-256 | trainable, 47.05M / 114.30M |
| Audio proj | MMAudioProjector | LN(512)→Linear→GELU→Linear(768) | trainable, 0.99M |
| Vision proj | MMVisionProjector | LN(768)→Linear→GELU→Linear(768) | trainable, 1.18M |
| Audio encoder | SenseVoice-Small | 50 blocks, output 512, 16kHz | frozen, 234.00M |
| Vision encoder | SigLIP2 base p32-256 | 12 layers, hidden 768, 64 tokens | frozen, 94.55M |
| Speech codec | Mimi | 8 codebooks, size 2048, 12.5Hz, 24kHz | frozen, 96.15M |
| Speaker cond. | CAM++ → spk_proj | 192-dim → hidden 768 | precomputed |

## 6. 训练曲线

### T2A 训练曲线（图 6）
- 使用清洁的日志段（排除了因加载不兼容 checkpoint 导致的 loss spike 区间）
- MoE 和 Dense 的收敛趋势接近

### A2A 训练曲线（图 7）
- 在 T2A 之后进行，暴露完整的 speech-in/speech-out 闭环
- 学习率高于 T2A 阶段的初始全模型训练

## 7. 面试要点

### 问：为什么训练要分这么多阶段？
**灾难性遗忘**问题。如果同时训练所有能力，模型可能会丢失已经学好的能力。分阶段训练：
- 先学最容易的 T2A（文本→语音）
- 再通过投影器冻结策略逐步引入语音输入和图像输入
- `audio_proj`/`vision_proj` 模式保证新模态的适配不会干扰已有参数

### 问：为什么外部编码器（SenseVoice、SigLIP2、Mimi）都冻结？
- **计算量**：SenseVoice 有 234M 参数，比整个 Thinker 还大，训不动
- **通用表征**：这些编码器已在海量数据上预训练，其表征是通用的
- **设计原则**：MLP 投影器负责域适配，编码器保持通用

### 问：全模型训练时 Talker 能直接从 Thinker 梯度中受益吗？
能，通过 bridge layer。Thinker 的梯度可以反向传播到 bridge 层，从而影响 Thinker 的表征。但是 Talker 的梯度不会影响 Thinker 的其他参数——这是 bridge 设计的关键优势：**语义学习受声学目标的轻微引导而非主导**。
