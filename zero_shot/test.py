"""Quick smoke test for Qwen3-VL to generate a hard negative description."""
import argparse
from typing import Any, Dict, List, Tuple

import torch
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_messages(video_url: str, prompt: str, max_frames: int, max_pixels: int) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_url,
                    "max_pixels": max_pixels,
                    "max_frames": max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_model(model_id: str, device: str) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor, str]:
    # Download weights via ModelScope; remote code is resolved through model_id.
    cache_dir = snapshot_download(model_id)
    device_map = "auto" if device == "auto" else device
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
    return model, processor, cache_dir


def generate(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_url: str,
    prompt: str,
    max_frames: int,
    max_pixels: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    device: str,
) -> str:
    messages = build_messages(video_url, prompt, max_frames, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )

    target_device = device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(target_device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample test for Qwen3-VL.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Model ID to load.")
    parser.add_argument(
        "--video",
        type=str,
        default="https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4",
        help="Video URL to test.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="生成一个与视频内容不符但看起来合理的描述，作为难负样本。",
        help="Prompt used for generation.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for inference: auto/cuda/cpu.")
    parser.add_argument("--max_frames", type=int, default=16, help="Max frames to sample from video.")
    parser.add_argument("--max_pixels", type=int, default=128 * 32 * 32, help="Max pixels for video processing.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate.")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling; default is greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature when sampling.")
    args = parser.parse_args()

    model, processor, _ = load_model(args.model, args.device)
    output = generate(
        model=model,
        processor=processor,
        video_url=args.video,
        prompt=args.prompt,
        max_frames=args.max_frames,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
