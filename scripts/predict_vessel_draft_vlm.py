import argparse
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.vessel_draft_dataset import build_vessel_draft_processor, calc_image_token_len
from trainer.trainer_utils import get_model_params, setup_seed
'''
cd /home/zby/projects/minimind-v

python scripts/predict_vessel_draft_vlm.py \
  --weight_path /home/zby/projects/minimind-v/out/test/vesseldraft_sft_768.pth \
  --data_path /home/zby/data/FullProcess/VesselDraftDeading/test_clean.jsonl \
  --output_path /home/zby/projects/minimind-v/out/test/vesseldraft_test_predictions.jsonl \
  --device cuda:0
'''

DEFAULT_PROMPT = "<image>\n请根据船舶吃水区域截图完成吃水读数估计。要求先定位全部刻度字符框，由刻度行右下角Y坐标推理相邻刻度透视步长；再定位水体上边界并统计水线Y；最后选择最接近水线且可用的完整刻度作为锚点，给出计算过程和最终吃水读数。"


def normalize_conversations(conversations):
    role_map = {'human': 'user', 'gpt': 'assistant'}
    normalized = []
    for turn in conversations:
        role = turn.get('role', turn.get('from'))
        content = turn.get('content', turn.get('value', ''))
        normalized.append({'role': role_map.get(role, role), 'content': content})
    return normalized


def get_user_prompt(row):
    for turn in normalize_conversations(row.get('conversations', [])):
        if turn.get('role') == 'user':
            return turn.get('content', DEFAULT_PROMPT)
    return DEFAULT_PROMPT


def get_target_answer(row):
    for turn in normalize_conversations(row.get('conversations', [])):
        if turn.get('role') == 'assistant':
            return turn.get('content', '')
    return ''


def resolve_image_path(data_dir, image_field):
    image_path = image_field[0] if isinstance(image_field, list) else image_field
    if not os.path.isabs(image_path):
        image_path = os.path.join(data_dir, image_path)
    return image_path


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        image_token_len=calc_image_token_len(args.image_height, args.image_width),
    )
    model = MiniMindVLM(config, vision_model_path=args.vision_model_path)
    state_dict = torch.load(args.weight_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)

    if args.device != 'cpu':
        if args.dtype == 'bfloat16':
            model = model.to(torch.bfloat16)
        elif args.dtype == 'float16':
            model = model.half()
    model = model.eval().to(args.device)
    model.processor = build_vessel_draft_processor(
        image_height=args.image_height,
        image_width=args.image_width,
        preserve_aspect_ratio=bool(args.preserve_aspect_ratio),
    )
    get_model_params(model, model.config)
    return model, tokenizer, model.processor


def build_inputs(model, tokenizer, prompt, device, max_seq_len, open_thinking):
    if '<image>' not in prompt:
        prompt = '<image>\n' + prompt
    image_tokens = model.config.image_special_token * model.config.image_token_len
    messages = [{'role': 'user', 'content': prompt.replace('<image>', image_tokens)}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(open_thinking),
    )
    return tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_seq_len).to(device)


def build_pixel_values(image_path, preprocess, device, pad_to_square):
    image = Image.open(image_path).convert('RGB')
    if pad_to_square:
        image = pad_image(image)
    return {k: v.to(device) for k, v in MiniMindVLM.image2tensor(image, preprocess).items()}


def predict_one(model, tokenizer, preprocess, image_path, prompt, args):
    inputs = build_inputs(model, tokenizer, prompt, args.device, args.max_seq_len, args.open_thinking)
    pixel_values = build_pixel_values(image_path, preprocess, args.device, bool(args.pad_to_square))
    attention_mask = inputs.get("attention_mask")
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        pixel_values=pixel_values,
    )
    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    stopped_by_eos = bool(len(new_tokens) > 0 and new_tokens[-1].item() == tokenizer.eos_token_id)
    return {
        "text": text,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(len(new_tokens)),
        "finish_reason": "eos" if stopped_by_eos else "length",
    }


def predict_jsonl(args, model, tokenizer, preprocess):
    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.data_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]

    with open(args.output_path, 'w', encoding='utf-8') as out:
        for idx, row in enumerate(rows, start=1):
            image_path = resolve_image_path(data_dir, row['image'])
            prompt = get_user_prompt(row)
            pred = predict_one(model, tokenizer, preprocess, image_path, prompt, args)
            result = {
                'id': row.get('id'),
                'split': row.get('split'),
                'image': row.get('image'),
                'prompt': prompt,
                'prediction': pred["text"],
                'input_tokens': pred["input_tokens"],
                'output_tokens': pred["output_tokens"],
                'finish_reason': pred["finish_reason"],
                'target': get_target_answer(row),
            }
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"[{idx}/{len(rows)}] {row.get('id', image_path)} ({pred['finish_reason']}, {pred['output_tokens']} tok) -> {pred['text'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="VesselDraftDeading VLM prediction")
    parser.add_argument("--weight_path", type=str, default=os.path.join(PROJECT_ROOT, "out/test/vesseldraft_sft_768.pth"), help="训练后的pth权重路径")
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.expanduser("~"), "data/FullProcess/VesselDraftDeading/test_clean.jsonl"), help="待预测jsonl路径")
    parser.add_argument("--output_path", type=str, default=os.path.join(PROJECT_ROOT, "out/test/vesseldraft_predictions.jsonl"), help="批量预测输出jsonl")
    parser.add_argument("--image_path", type=str, default="", help="单张图片路径；传入后忽略data_path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="单张图片推理prompt")
    parser.add_argument("--tokenizer_path", type=str, default=os.path.join(PROJECT_ROOT, "model"), help="tokenizer目录")
    parser.add_argument("--vision_model_path", type=str, default=os.path.join(PROJECT_ROOT, "model/siglip2-base-p32-256-ve"), help="视觉编码器目录")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=4096, type=int, help="最大输入长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE权重")
    parser.add_argument("--max_new_tokens", default=3200, type=int, help="最大生成token数")
    parser.add_argument("--temperature", default=0.7, type=float, help="采样温度")
    parser.add_argument("--top_p", default=0.85, type=float, help="top-p采样")
    parser.add_argument("--do_sample", default=0, type=int, choices=[0, 1], help="0=贪心解码，1=采样")
    parser.add_argument("--repetition_penalty", default=1.08, type=float, help="重复惩罚，降低循环输出")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="推理精度")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="推理设备")
    parser.add_argument("--limit", default=0, type=int, help="只预测前N条，0表示全部")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument("--open_thinking", default=0, type=int, help="是否开启chat template thinking")
    parser.add_argument("--pad_to_square", default=0, type=int, choices=[0, 1], help="是否先把长图补成正方形")
    parser.add_argument("--image_height", default=256, type=int, help="送入视觉编码器的图像高度")
    parser.add_argument("--image_width", default=640, type=int, help="送入视觉编码器的图像宽度")
    parser.add_argument("--preserve_aspect_ratio", default=1, type=int, choices=[0, 1], help="是否等比例缩放并补边")
    args = parser.parse_args()

    setup_seed(args.seed)
    model, tokenizer, preprocess = load_model(args)
    if args.image_path:
        pred = predict_one(model, tokenizer, preprocess, args.image_path, args.prompt, args)
        print(pred["text"])
        print(f"\nfinish_reason={pred['finish_reason']}, input_tokens={pred['input_tokens']}, output_tokens={pred['output_tokens']}")
    else:
        predict_jsonl(args, model, tokenizer, preprocess)
        print(f"预测结果已保存: {args.output_path}")


if __name__ == "__main__":
    main()
