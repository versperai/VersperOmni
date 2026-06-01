# Pretrain Data

The pretraining data for the current main branch is `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`.
These two datasets have been organized into a unified `text -> next token prediction` training format, aiming to balance under limited compute.


## 1. Core Optimization Pillars

- Text quality;
- Length distribution;
- Chinese-English mixed capability;
- Template alignment with subsequent SFT / Tool Calling / RLAIF stages.


## 2. Data Sourcing & Curation Pipeline

* **Premium Data Sources** (Governed under *permissive open-source licenses*):
    * `General Text`: Broad-scale corpora for factual base scaffolding.
    * `Curated Dialogue`: High-quality structural conversational interactions.
    * `Distillation Corpora`: Advanced synthetic knowledge derived from large models.
    * `Public Repositories`: Prominent community datasets including [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data) and [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align).
* **Four-Stage Curation Protocol** (Mandatory before training entry):
    1.  **Cleaning**: Removing low-quality fragments and irrelevant formatting.
    2.  **Deduplication**: Eliminating repetitive text to prevent overfitting.
    3.  **Length Control**: Enforcing strict upper and lower sequence boundaries.
    4.  **Format Unification**: Conforming diverse data structures into a unified text schema.

- `pretrain_t2t_mini.jsonl` is intended for quick reproduction;
- `pretrain_t2t.jsonl` is intended for full training of the main branch model.


## 3. The File Format

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```
