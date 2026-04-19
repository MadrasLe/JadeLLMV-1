"""
Minimal local inference CLI for Jade-family models on Hugging Face.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference with a Jade model.")
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. Madras1/Jade-20B")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--system", default=None, help="Optional system instruction")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def build_prompt(tokenizer, user_prompt: str, system_prompt: str | None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if system_prompt:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    return f"User: {user_prompt}\nAssistant:"


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=args.trust_remote_code)

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
    }
    torch_dtype = resolve_dtype(args.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.repo, **model_kwargs)

    prompt_text = build_prompt(tokenizer, args.prompt, args.system)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
