# VersperOmni 开发文档

> 基于 [MiniMind-O Technical Report](https://github.com/jingyaogong/minimind-o) 整理
> 用于学习和面试准备，以中文编写

## 文档索引

| 文档 | 内容 | 阅读时长 |
|------|------|---------|
| [01-overview.md](01-overview.md) | 整体架构总览：Thinker-Talker、模块配置、数据流 | 15 min |
| [02-thinker-talker.md](02-thinker-talker.md) | Thinker-Talker 机制详解：架构细节、桥接、低秩接口 | 20 min |
| [03-data-format.md](03-data-format.md) | 序列格式与流式解码：9 流序列、损失函数、延时对齐 | 15 min |
| [04-training.md](04-training.md) | 训练流程详解：5 阶段训练、超参数、模块配置 | 15 min |
| [05-design-choices.md](05-design-choices.md) | 关键设计决策：桥接层选择、低秩接口、说话人控制 | 20 min |
| [06-performance.md](06-performance.md) | 性能与实验结果：消融研究、跨模型对比 | 15 min |

## 概念关系图

```
研究动机: 在 0.1B 规模下实现可复现的全模态闭环
         │
         ▼
核心架构: Thinker-Talker 分离设计
         │
         ├── Thinker → MiniMind Transformer（语义路径）
         ├── Talker → 4 层模块（声学路径）
         ├── Bridge → 中层 hidden state 桥接
         │
         ▼
多模态输入:
         ├── Audio → SenseVoice(冻结) → MLP Projector
         ├── Image → SigLIP2(冻结) → MLP Projector
         └── Speaker → CAM++ Embedding(预计算)
         │
         ▼
数据格式: 9 流序列（1 文本 + 8 codec）参考提示 + 延时对齐
         │
         ▼
训练策略: 5 阶段递进训练（全模型 ↔ 投影器冻结交替）
         │
         ▼
输出管线: Talker → 8 Codebook Heads → Mimi Decoder → 24kHz Waveform
```

## 代码结构

```
src/
├── model/
│   ├── versper.py          # MiniMind (Thinker) 基础模型
│   ├── versper_vl.py       # MiniMind-V (视觉) 扩展
│   ├── versper_omni.py     # MiniMind-O (全模态) 扩展
│   └── module/
│       ├── attention.py    # 注意力机制实现
│       └── norm.py         # 归一化层
├── evaluator/
│   ├── config.py           # 评估配置
│   ├── metrics.py          # 评估指标
│   └── ppl_eval.py         # PPL 评估器
└── main.py                 # 入口
```

> **注意**：当前代码库处于早期阶段，大部分模块为占位模板。
> VersionOmni 旨在复现 MiniMind-O 架构，文档基于技术报告先行编写。
