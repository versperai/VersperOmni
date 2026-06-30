"""
Simple text chat demo using VersperOmni LM.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer
from config import MiniMindConfig
from models.llm import MiniMindForCausalLM


def main():
    config = MiniMindConfig()
    model = MiniMindForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("./model")

    # Load weights
    state_dict = torch.load("./out/pretrain_768.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    while True:
        prompt = input("\nUser: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        out = model.generate(input_ids, max_new_tokens=1024, temperature=0.85, top_p=0.85)
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
