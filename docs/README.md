# VersperOmni 开发文档

> 基于 [MiniMind-O Technical Report](https://github.com/jingyaogong/minimind-o) 整理的学习/面试资料
> 统一代码库位于 `src/versper/`，涵盖 MiniMind（文本 LM）、MiniMind-V（视觉语言模型）、MiniMind-O（全模态模型）

原 MiniMind-O 技术报告可查阅 [`docs/archive/`](archive/) 目录。

---

## 文档导航

| 分类 | 文档 | 内容 |
|------|------|------|
| 架构概念 | [architecture/01-overview.md](architecture/01-overview.md) | 整体架构总览 |
| 架构概念 | [architecture/02-thinker-talker.md](architecture/02-thinker-talker.md) | Thinker-Talker 机制详解 |
| 架构概念 | [architecture/03-data-format.md](architecture/03-data-format.md) | 9 流序列格式与流式解码 |
| 架构概念 | [architecture/04-design-choices.md](architecture/04-design-choices.md) | 关键设计决策分析 |
| 训练指南 | [training/01-pretrain.md](training/01-pretrain.md) | 预训练数据与流程 |
| 训练指南 | [training/02-sft.md](training/02-sft.md) | SFT 数据与流程 |
| 训练指南 | [training/03-rl.md](training/03-rl.md) | RL/DPO 数据 |
| 训练指南 | [training/04-omni-strategy.md](training/04-omni-strategy.md) | Omni 5 阶段训练策略 |
| 代码解析 | [code/01-config.md](code/01-config.md) | 配置体系 (Config) |
| 代码解析 | [code/02-lm.md](code/02-lm.md) | 核心语言模型 |
| 代码解析 | [code/03-vlm.md](code/03-vlm.md) | 视觉语言模型 |
| 代码解析 | [code/04-omni.md](code/04-omni.md) | 全模态模型 |
| 代码解析 | [code/05-datasets.md](code/05-datasets.md) | 数据集实现 |
| 代码解析 | [code/06-training-scripts.md](code/06-training-scripts.md) | 训练脚本 |
| 实战指南 | [guides/00-setup.md](guides/00-setup.md) | 环境搭建与安装 |
| 实战指南 | [guides/01-inference.md](guides/01-inference.md) | 快速推理指南 |
| 实战指南 | [guides/02-demo.md](guides/02-demo.md) | 交互式演示与 Web Demo |
| 实战指南 | [guides/03-evaluation.md](guides/03-evaluation.md) | 模型评估指南 |
| 实战指南 | [guides/04-advanced.md](guides/04-advanced.md) | 高级特性（流式/VAD/实时会话） |
| 实战指南 | [guides/05-tui.md](guides/05-tui.md) | TUI 终端界面 |
| 参考速查 | [reference/01-performance.md](reference/01-performance.md) | 性能与实验结果 |
| 参考速查 | [reference/02-interview-qa.md](reference/02-interview-qa.md) | 面试问答集锦 |

---

## 架构概念图

```
Text ────────────► ┌─────────────────────┐
                    │      Thinker         │────► Text Output
Audio ─►SenseVoice─┤   (MiniMind 8层)    │
        (frozen)   │                     │
Image ─► SigLIP2 ──┤  bridge layer 3     ├────► bridge state
         (frozen)  └─────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │       Talker        │
Codec History ─────►│   (4 MiniMind层)    ├────► Mimi Codes ──► 24kHz Audio
Speaker Ref ───────►│  fusion: bridge +   │
                    │  codec history      │
                    └─────────────────────┘
```

## 代码结构总览

```
src/versper/
├── __init__.py          # 统一导出 MiniMind / VLM / Omni
├── config.py            # 配置体系：MiniMindConfig → VLMConfig → OmniConfig
├── model.py             # MiniMind 基础语言模型 (Thinker)
├── vlm.py               # MiniMind-V 视觉语言模型扩展
├── omni.py              # MiniMind-O 全模态模型 (Thinker + Talker)
├── modules/             # 子模块（注意力、归一化等）
├── dataset/
│   ├── lm_dataset.py    # 文本预训练/SFT 数据集
│   ├── vlm_dataset.py   # 视觉语言数据集
│   └── omni_dataset.py  # 全模态数据集（T2A, I2T, A2A）
├── trainer/
│   ├── pretrain.py      # 预训练脚本
│   ├── sft_vlm.py       # VLM SFT 脚本
│   ├── sft_omni.py      # Omni SFT 脚本
│   └── utils.py         # 训练工具
└── scripts/
    └── web_demo.py      # Web 交互演示
```

### 关键导入路径

```python
# 文本/LM
from versper.model import MiniMindForCausalLM
from versper.config import MiniMindConfig

# 视觉语言 (VLM)
from versper.vlm import MiniMindVLM
from versper.config import VLMConfig

# 全模态 (Omni)
from versper.omni import MiniMindOmni
from versper.config import OmniConfig
```

### 训练启动方式

```bash
# 文本预训练
python -m versper.trainer.pretrain

# VLM SFT
python -m versper.trainer.sft_vlm

# Omni SFT（支持 --mode all / thinker / talker）
python -m versper.trainer.sft_omni --mode all
```

## 关联说明

- 本仓库统一了三个模型变体：**MiniMind**（纯文本）、**MiniMind-V**（文本+图像输入）、**MiniMind-O**（文本+语音+图像输入，语音输出）
- 配置体系采用继承链：`MiniMindConfig → VLMConfig → OmniConfig`
- 各文档中的代码示例均使用 `src/versper/` 下的新路径，与旧版 `src/model/` 路径不同
