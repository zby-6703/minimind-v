import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_vlm import MiniMindVLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


SYSTEM_PROMPTS = [
    "你是一个专业的船舶吃水读数助手，请根据图像和几何线索给出准确推理。",
    "你是一个可靠的视觉语言助手，请严格按照用户要求完成船舶吃水估计。",
]


def calc_image_token_len(image_height: int, image_width: int, patch_size: int = 32) -> int:
    return max(1, image_height // patch_size) * max(1, image_width // patch_size)


class VesselDraftImageProcessor:
    def __init__(
        self,
        image_height=256,
        image_width=640,
        preserve_aspect_ratio=True,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def _resize_and_pad(self, image):
        if not self.preserve_aspect_ratio:
            return image.resize((self.image_width, self.image_height), Image.Resampling.BICUBIC)
        width, height = image.size
        scale = min(self.image_width / width, self.image_height / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (self.image_width, self.image_height), (0, 0, 0))
        left = (self.image_width - resized_w) // 2
        top = (self.image_height - resized_h) // 2
        canvas.paste(image, (left, top))
        return canvas

    def _to_tensor(self, image):
        image = image.convert('RGB')
        image = self._resize_and_pad(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.image_mean) / self.image_std

    def __call__(self, images, return_tensors="pt", **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = torch.stack([self._to_tensor(img) for img in images], dim=0)
        return {"pixel_values": pixel_values}


def build_vessel_draft_processor(image_height=256, image_width=640, preserve_aspect_ratio=True):
    return VesselDraftImageProcessor(image_height=image_height, image_width=image_width, preserve_aspect_ratio=preserve_aspect_ratio)


class VesselDraftDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        preprocess=None,
        max_length=1024,
        image_special_token='<|image_pad|>',
        image_token_len=64,
        add_system_ratio=0.0,
        pad_to_square=False,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.data_dir = os.path.dirname(os.path.abspath(jsonl_path))
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.add_system_ratio = add_system_ratio
        self.pad_to_square = pad_to_square
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def normalize_conversations(conversations):
        role_map = {'human': 'user', 'gpt': 'assistant'}
        normalized = []
        for turn in conversations:
            role = turn.get('role', turn.get('from'))
            content = turn.get('content', turn.get('value', ''))
            normalized.append({'role': role_map.get(role, role), 'content': content})
        return normalized

    def maybe_add_system(self, conversations):
        if not conversations or conversations[0].get('role') == 'system':
            return conversations
        if self.add_system_ratio > 0 and random.random() < self.add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
        return conversations

    def resolve_image_paths(self, image_field):
        if not isinstance(image_field, list):
            image_field = [image_field]
        paths = []
        for image_path in image_field:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_dir, image_path)
            paths.append(image_path)
        return paths

    def pad_image(self, image):
        if not self.pad_to_square:
            return image
        w, h = image.size
        if w == h:
            return image
        side = max(w, h)
        canvas = Image.new('RGB', (side, side), (0, 0, 0))
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))
        return canvas

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content']
            if turn.get('role') != 'system':
                content = content.replace('<image>', self.image_special_token)
            messages.append({'role': turn['role'], 'content': content})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        row = self.rows[index]
        conversations = self.normalize_conversations(row['conversations'])
        conversations = self.maybe_add_system(conversations)
        prompt = self.create_chat_prompt(conversations)

        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_inputs = []
        for image_path in self.resolve_image_paths(row['image']):
            image = self.pad_image(Image.open(image_path).convert('RGB'))
            image_inputs.append(MiniMindVLM.image2tensor(image, self.preprocess))

        if hasattr(image_inputs[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs], dim=0) for k in image_inputs[0].keys()}
        else:
            image_data = torch.stack(image_inputs)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data
