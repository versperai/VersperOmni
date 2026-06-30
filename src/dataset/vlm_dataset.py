"""
Dataset for Vision-Language Model training (I2T).
Consolidated from minimind-v/dataset/lm_dataset.py
"""
import os
import json
import random
import io
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from models.vlm import VersperVLM
from .lm_dataset import pre_processing_chat, post_processing_chat

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess=None,
        max_length=512,
        image_special_token="<|image_pad|>",
        image_token_len=64,
    ):
        super().__init__()
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.dataset)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = (
                turn["content"].replace("<image>", self.image_special_token)
                if turn.get("role") != "system"
                else turn["content"]
            )
            messages.append({"role": turn["role"], "content": content})
        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        row = self.dataset[index]
        conversations = (
            json.loads(row["conversations"])
            if isinstance(row["conversations"], str)
            else row["conversations"]
        )
        image_bytes = row["image_bytes"]
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]

        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels = self.generate_labels(input_ids)

        image_inputs_list = [
            VersperVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess)
            for img in image_bytes
        ]
        if hasattr(image_inputs_list[0], "keys"):
            image_data = {
                k: torch.cat([inp[k] for inp in image_inputs_list], dim=0)
                for k in image_inputs_list[0].keys()
            }
        else:
            image_data = torch.stack(image_inputs_list)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_data,
        )


def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], "keys"):
        pixel_values = {
            k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()
        }
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values
