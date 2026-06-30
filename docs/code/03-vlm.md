# 视觉语言模型 -- VersperVLM

## 概述

`VersperVLM` (`vlm.py:34`) 继承自 `VersperForCausalLM`，在纯文本语言模型的基础上添加了视觉输入能力。它使用 SigLIP2 作为视觉编码器，通过一个 MLP 投影器将视觉特征映射到 LLM 的隐空间。

```
VersperForCausalLM
  └── VersperVLM (config_class = VLMConfig)
        ├── vision_encoder: SigLIP2 (冻结, 94.55M 参数)
        ├── processor: SiglipImageProcessor
        └── vision_proj: MMVisionProjector (1.18M 参数)
```

---

## 架构组件

### MMVisionProjector

定义在 `vlm.py:18`，是一个 2 层 MLP：

```python
MMVisionProjector(768 → 768):
  LayerNorm(768)
  → Linear(768, 768)
  → GELU
  → Linear(768, 768)
```

参数量: `768×2 + 768×768 + 768×768 = 1,180,416 ≈ 1.18M`

### SigLIP2 视觉编码器

| 属性 | 值 |
|------|-----|
| 模型 | SigLIP2 base patch32-256 |
| 参数 | ~94.55M |
| Transformer 层数 | 12 |
| Patch 大小 | 32×32 |
| 输入分辨率 | 256×256 |
| 输出特征维度 | 768 |
| 每张图片 token 数 | 64 |
| 是否冻结 | 是 (`requires_grad = False`) |

### 处理器

使用 HuggingFace `SiglipImageProcessor` 进行图像预处理，包括调整大小、归一化等。

---

## 初始化

```python
class VersperVLM(VersperForCausalLM):
    def __init__(
        self,
        config: VLMConfig = None,
        vision_model_path="./model/siglip2-base-p32-256-ve",
    ):
```

- 如果 `vision_model_path` 不存在或加载失败，`vision_encoder` 和 `processor` 均为 `None`
- 这种情况下模型退化为纯文本模型，等同于 `VersperForCausalLM`

### 无视觉路径的 VLM

```python
# 仅加载 LLM 权重，无视觉编码器
vlm = VersperVLM(config, vision_model_path="nonexistent")
assert vlm.vision_encoder is None  # 退化为文本模型
```

### 完整 VLM

```python
vlm_config = VLMConfig(hidden_size=768, num_hidden_layers=8)
vlm = VersperVLM(vlm_config, vision_model_path="./model/siglip2-base-p32-256-ve")
```

---

## 视觉特征注入

核心方法 `_inject_vision_features` (`vlm.py:85`) 负责将图像特征注入到文本 embedding 序列中。

### 注入流程

```python
# 1. 文本 token embedding (常规)
hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

# 2. 编码图像 (仅在首次前向，start_pos == 0)
if pixel_values is not None and start_pos == 0:
    # 2a. SigLIP2 编码 → image embeddings
    image_outputs = self.vision_encoder(**pixel_values)
    vision_tensors = self.vision_proj(image_outputs.last_hidden_state)
    
    # 2b. 替换占位符 token
    hidden_states = self._inject_vision_features(
        tokens=input_ids,
        h=hidden_states,
        vision_tensors=vision_tensors,
        seqlen=input_ids.shape[1],
    )
```

### 占位符替换逻辑

`_inject_vision_features` 逐 batch 扫描 token 序列：

```
输入序列: [token_0, token_1, ..., <|image_pad|>(x64), ..., token_N]
                                    ↓
替换后:   [token_0, token_1, ..., vision_token_0..63, ..., token_N]
```

- 占位符 token ID: `config.image_ids[0]` (默认 12)
- 连续相同占位符的数量决定每张图片的 token 数
- 支持多张图片，通过 `vision_tensors` 的第二维区分

### 多图片支持

```python
# pixel_values shape: (batch, num_images, 3, 256, 256)
# vision_tensors shape: (batch, num_images, 64, 768)
```

---

## 前向传播

`VersperVLM.forward` (`vlm.py:114`) 扩展了父类的 forward：

1. **文本 embedding**: 同父类
2. **视觉注入**: 仅在 `start_pos == 0` 时执行
3. **Transformer 层**: 同父类，逐层通过 `self.model.layers`
4. **LM Head**: 同父类
5. **损失计算**: 同父类，CrossEntropy(ignore_index=-100)

视觉注入只在预填充阶段（首次前向）发生，后续自回归解码阶段的 `past_key_values` 已经包含了视觉信息。

### DDP 兼容性

```python
aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
```

这行代码的目的是在 DDP 训练中为 `vision_proj` 的参数创建计算图依赖，避免 DDP 因某些参数未参与 loss 计算而报错。

---

## 生成

`generate` 方法 (`vlm.py:235`) 重写父类以处理 `pixel_values` 的多序列复制：

```python
def generate(self, *args, num_return_sequences=1, **kwargs):
    if num_return_sequences > 1 and "pixel_values" in kwargs:
        # 复制 pixel_values 以匹配多序列
        kwargs["pixel_values"] = pv.repeat(num_return_sequences, ...)
    return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
```

---

## 使用示例

```python
from PIL import Image
from versper.config import VLMConfig
from versper.vlm import VersperVLM

# 初始化
config = VLMConfig()
model = VersperVLM(config, vision_model_path="./model/siglip2-base-p32-256-ve")
model.eval()

# 图像预处理
image = Image.open("photo.jpg")
pixel_values = VersperVLM.image2tensor(image, model.processor)

# 推理
input_ids = tokenizer.encode("<|image_pad|>这是什么？")
output = model.generate(
    input_ids,
    pixel_values=pixel_values,
    max_new_tokens=128,
)
```
