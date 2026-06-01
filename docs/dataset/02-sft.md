# SFT Data

The SFT data for the current main branch is `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`. Compared with earlier `sft_512 / sft_1024 / sft_2048` schemes, the current version places more emphasis on:

- Unified templates;
- Better suited for mixed training of dialogue + thinking tags + Tool Calling;
- Minimizing data preprocessing forks, reducing reproduction costs.

Its data sources include but are not limited to high-quality instruction-following data, public dialogue data, model-distilled synthetic data, and permissively licensed open-source datasets; before entering the `t2t` main branch, they are unified into the multi-turn dialogue format used by the current repository. The current main branch also contains a large amount of synthetic data, such as approximately `100K` `tool call` entries synthesized from `qwen3-4b`, as well as `reasoning` data from the `qwen3` series. Major community sources include: [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align), [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT), [COIG](https://huggingface.co/datasets/BAAI/COIG), [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT), etc. Published versions ensure that data sources and processing pipelines comply with the transitivity constraints of corresponding open-source licenses, and adhere to Apache-2.0, CC-BY-NC-2.0, and other related license requirements.

Among them:

- `sft_t2t_mini.jsonl`: suitable for quickly training a dialogue model;
- `sft_t2t.jsonl`: suitable for fully reproducing the main branch version;
- `toolcall` capability has already been merged into the main branch SFT data.

All SFT files follow the same format, including dialogue and Tool Use data:

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```



