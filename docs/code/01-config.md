# 配置体系详解 -- VersperConfig → VLMConfig → OmniConfig

## 概述

VersperOmni 采用三级配置继承结构，从纯文本模型逐步扩展到全模态模型。所有配置类均继承自 HuggingFace `PretrainedConfig`，因此与 Transformers 生态（`from_pretrained` / `save_pretrained`）完全兼容。

```
PretrainedConfig
  └── VersperConfig    (纯文本 LM)
        └── VLMConfig   (+ 视觉)
              └── OmniConfig  (+ 音频 + Talker)
```

三个配置类分别对应 `model_type`：
- `VersperConfig` → `"versper"`
- `VLMConfig` → `"versper-v"`
- `OmniConfig` → `"versper-o"`

---

## VersperConfig (文本基类)

定义在 `config.py:9`，是全部变体的共同基础。

### 关键参数及默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 6400 | 词表大小 |
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | 查询头数 (Q heads) |
| `num_key_value_heads` | 4 | 键值头数 (KV heads)，GQA 关键参数 |
| `head_dim` | 96 (768/8) | 每头维度 |
| `intermediate_size` | ceil(768×π/64)×64 = 768 | FFN 中间层维度，π 缩放 |
| `max_position_embeddings` | 32768 | 最大位置编码长度 |
| `dropout` | 0.0 | Dropout 比率 |
| `flash_attn` | True | 是否启用 Flash Attention |
| `hidden_act` | "silu" | 激活函数 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 1e6 | RoPE base frequency |
| `tie_word_embeddings` | True | 是否绑定输入输出词嵌入 |
| `inference_rope_scaling` | False | 推理时是否启用 YaRN |
| `rope_scaling` | None / dict | YaRN 参数（beta_fast=32, beta_slow=1, factor=16）|

### Intermediate Size 的计算

```python
intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
```

以 hidden_size=768 为例：`ceil(768×3.14159/64) = ceil(37.699) = 38`，`38×64 = 2432`。

### MoE 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_moe` | False | 是否启用 MoE |
| `num_experts` | 4 | 专家总数 |
| `num_experts_per_tok` | 1 | 每 token 激活的专家数 |
| `moe_intermediate_size` | = intermediate_size | MoE 专家的 intermediate size |
| `norm_topk_prob` | True | 是否对 top-k 权重做归一化 |
| `router_aux_loss_coef` | 5e-4 | 路由辅助损失系数 |

### 使用示例

```python
from versper.config import VersperConfig

# Dense 模型 (8层, 768维)
config = VersperConfig()

# 自定义配置
config = VersperConfig(
    hidden_size=512,
    num_hidden_layers=6,
    vocab_size=32000,
    max_position_embeddings=8192,
)

# MoE 模型
config = VersperConfig(
    use_moe=True,
    num_experts=8,
    num_experts_per_tok=2,
)
```

---

## VLMConfig (视觉扩展)

定义在 `config.py:55`，继承 `VersperConfig`，添加视觉相关参数。

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_special_token` | `"<\|image_pad\|>"` | 图像占位符文本 |
| `image_ids` | [12] | 图像占位符 token ID |
| `image_hidden_size` | 768 | SigLIP2 视觉编码器输出维度 |
| `image_token_len` | 64 | 每张图片分配的 token 数 |

### 使用示例

```python
from versper.config import VLMConfig

vlm_config = VLMConfig(
    hidden_size=768,
    num_hidden_layers=8,
    image_token_len=64,
)
```

---

## OmniConfig (全模态扩展)

定义在 `config.py:66`，继承 `VLMConfig`，添加音频和 Talker 相关参数。

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_talker_hidden_layers` | 4 | Talker 模块的 Transformer 层数 |
| `talker_hidden_size` | 768 | Talker 隐藏层维度 |
| `audio_ids` | [16] | 音频占位符 token ID |
| `audio_special_token` | `"<\|audio_pad\|>"` | 音频占位符文本 |
| `audio_hidden_size` | 512 | SenseVoice 编码器输出维度 |
| `audio_feat_dim` | 512 | 音频特征维度 |
| `audio_vocab_size` | 2112 | Mimi Codec 码本大小 |
| `audio_num_codebooks` | 8 | 音频码本数量（Mimi Codec 层数）|
| `audio_pad_token` | 2049 | 音频填充 token ID |
| `audio_stop_token` | 2050 | 音频停止 token ID |
| `audio_spk_token` | 2051 | 说话人 token ID |
| `spk_emb_size` | 192 | 说话人嵌入维度 |
| `think_end_ids` | [26, 234, 234] | `"</think>\n\n"` 对应的 token IDs |
| `bridge_layer` | None → 自动计算 | Thinker→Talker 桥接层 |

### Bridge Layer 的自动计算

桥接层用于从 Thinker 输出的中间层 hidden states 连接到 Talker 模块。其默认值在 `super().__init__()` 之后设置，因为它依赖于 `num_hidden_layers`：

```python
class OmniConfig(VLMConfig):
    def __init__(self, **kwargs):
        # 先收集自己的参数
        self.bridge_layer = kwargs.pop("bridge_layer", None)
        # ...
        super().__init__(**kwargs)
        # 父类初始化完成后才计算
        if self.bridge_layer is None:
            self.bridge_layer = self.num_hidden_layers // 2 - 1
```

以 `num_hidden_layers=8` 为例：`bridge_layer = 8//2 - 1 = 3`（即第 4 层，从 0 开始计数）。

### 使用示例

```python
from versper.config import OmniConfig

# 默认 Omni 配置 (8层 thinker, 4层 talker)
omni_config = OmniConfig()

# 更大规模的配置
omni_config = OmniConfig(
    hidden_size=1024,
    num_hidden_layers=12,
    num_talker_hidden_layers=6,
    talker_hidden_size=1024,
    use_moe=True,
)

# 访问关键参数
print(omni_config.bridge_layer)  # 5 (12//2 - 1)
print(omni_config.audio_vocab_size)  # 2112
print(omni_config.audio_num_codebooks)  # 8
```

---

## 选择 Dense vs MoE

通过 `use_moe=True` 即可启用 MoE。MoE 模式下：
- `VersperBlock` 中的 `self.mlp` 将使用 `MOEFeedForward` 而非 `FeedForward`
- 模型输出中包含 `aux_loss`，用于负载均衡
- 训练时 `router_aux_loss_coef` 控制辅助损失权重

```python
# Dense
config = VersperConfig(use_moe=False)

# MoE (4 experts, top-1)
config = VersperConfig(
    use_moe=True,
    num_experts=4,
    num_experts_per_tok=1,
)
```
