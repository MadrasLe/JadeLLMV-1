"""
Supervised fine-tuning entrypoint for Jade-family chat models.

This script is meant to be both runnable and readable:

- it accepts multiple common JSONL conversation formats,
- normalizes them into chat messages,
- trains with LoRA, DoRA, or full fine-tuning,
- and optionally merges / pushes the final artifact.

Example:
    python train_jade_sft.py --datasets data.jsonl --model unsloth/Qwen2.5-3B --adapter lora
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

from datasets import Dataset


DEFAULT_MODEL = "unsloth/Qwen2.5-3B"
DEFAULT_OUTPUT = "outputs/jade-sft"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Jade model with Unsloth SFT.")
    parser.add_argument("--datasets", nargs="+", required=True, help="One or more JSONL paths or globs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--adapter", choices=["lora", "dora", "full"], default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--push-to-hub", default=None, help="Destination repo id on Hugging Face Hub")
    parser.add_argument("--save-merged", action="store_true", help="Save a merged full model after training")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the output dir")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def expand_dataset_patterns(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if matches:
            resolved.extend(matches)
            continue
        direct_path = Path(pattern)
        if direct_path.exists():
            resolved.append(direct_path)
        else:
            print(f"[warn] dataset pattern not found: {pattern}")
    return sorted({path.resolve() for path in resolved})


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    invalid_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines += 1
    if invalid_lines:
        print(f"[warn] {path.name}: ignored {invalid_lines} invalid JSONL lines")
    return items


def normalize_messages(item: dict) -> dict | None:
    if "messages" in item:
        messages = [message for message in item["messages"] if message.get("role") != "system"]
        roles = [message.get("role") for message in messages]
        if messages and "user" in roles and "assistant" in roles:
            return {"messages": messages}
        return None

    if "instruction" in item:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\n\n{item['input']}"
        assistant_content = item.get("output") or item.get("response")
        if assistant_content:
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        return None

    if "question" in item or "prompt" in item:
        question = item.get("question") or item.get("prompt")
        answer = item.get("answer") or item.get("response") or item.get("output")
        if question and answer:
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
    return None


def load_training_dataset(dataset_paths: list[Path]) -> Dataset:
    raw_items: list[dict] = []
    for path in dataset_paths:
        loaded = load_jsonl(path)
        print(f"[data] {path.name}: {len(loaded)} raw examples")
        raw_items.extend(loaded)

    if not raw_items:
        raise ValueError("no training examples were loaded")

    normalized: list[dict] = []
    skipped = 0
    for item in raw_items:
        normalized_item = normalize_messages(item)
        if normalized_item is None:
            skipped += 1
            continue
        normalized.append(normalized_item)

    if not normalized:
        raise ValueError("all examples were invalid after normalization")

    print(f"[data] valid examples: {len(normalized)}")
    if skipped:
        print(f"[data] skipped examples: {skipped}")

    print("[data] sample conversation:")
    for message in normalized[0]["messages"]:
        preview = message["content"].replace("\n", " ")[:100]
        print(f"  - {message['role']}: {preview}...")

    return Dataset.from_list(normalized)


def import_training_stack():
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError as exc:
        raise RuntimeError("unsloth is required. Install it before running this script.") from exc

    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    return FastLanguageModel, get_chat_template, TrainingArguments, SFTTrainer, torch


def load_model(args: argparse.Namespace, FastLanguageModel, torch):
    print_section("Loading model")
    if args.adapter == "full":
        print("[model] full fine-tuning mode")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )
        if args.lr == 2e-4:
            args.lr = 2e-5
            print(f"[model] adjusted learning rate for full fine-tuning: {args.lr}")
        return model, tokenizer

    print(f"[model] adapter mode: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=list(DEFAULT_TARGET_MODULES),
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_dora=args.adapter == "dora",
    )
    return model, tokenizer


def apply_chat_template(dataset: Dataset, tokenizer, get_chat_template):
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def format_examples(batch):
        texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(format_examples, batched=True)
    lengths = [len(text) for text in formatted_dataset["text"]]
    print(f"[data] avg chars: {sum(lengths) // len(lengths):,}")
    print(f"[data] min chars: {min(lengths):,}")
    print(f"[data] max chars: {max(lengths):,}")
    return formatted_dataset, tokenizer


def build_training_args(args: argparse.Namespace, TrainingArguments, torch):
    return TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
    )


def find_resume_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))[-1]
    return str(latest)


def save_adapter(output_dir: Path, model, tokenizer):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[save] adapter saved to {output_dir}")


def merge_adapter_model(output_dir: Path, target_repo: str | None, save_merged: bool, torch):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_section("Merging adapter into base model")
    adapter_config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = adapter_config["base_model_name_or_path"]
    print(f"[merge] base model: {base_model_name}")

    # Reloading the base model in bf16 avoids the bad merge path you get when
    # trying to merge directly from a 4-bit training load.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    merged_model = PeftModel.from_pretrained(base_model, str(output_dir), torch_dtype=torch.bfloat16)
    merged_model = merged_model.merge_and_unload()

    merged_dir = output_dir.with_name(output_dir.name + "-merged")
    if save_merged:
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[merge] merged model saved to {merged_dir}")

    if target_repo:
        print(f"[hub] pushing merged model to {target_repo}")
        merged_model.push_to_hub(target_repo, safe_serialization=True, max_shard_size="4GB")
        tokenizer.push_to_hub(target_repo)

    del merged_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def push_training_output(args: argparse.Namespace, model, tokenizer, torch):
    output_dir = Path(args.output)
    if args.adapter == "full" and args.push_to_hub:
        print(f"[hub] pushing full model to {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(args.push_to_hub)
        return

    if args.adapter != "full" and (args.save_merged or args.push_to_hub):
        merge_adapter_model(output_dir, args.push_to_hub, args.save_merged, torch)
        return

    if args.push_to_hub:
        print(f"[hub] pushing adapter to {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


def print_summary(args: argparse.Namespace, trainer_stats, dataset_size: int):
    print_section("Training complete")
    print(f"[result] output: {args.output}")
    print(f"[result] train runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"[result] train loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"[result] dataset size: {dataset_size}")
    if args.save_merged:
        print(f"[result] merged output: {args.output}-merged")
    if args.push_to_hub:
        print(f"[result] hub repo: https://huggingface.co/{args.push_to_hub}")


def main():
    args = parse_args()

    print_section("Jade SFT")
    print(f"[config] model: {args.model}")
    print(f"[config] adapter: {args.adapter}")
    print(f"[config] epochs: {args.epochs}")
    print(f"[config] batch: {args.batch_size} x {args.gradient_accumulation}")
    print(f"[config] max_seq_length: {args.max_seq_length}")
    print(f"[config] output: {args.output}")

    dataset_paths = expand_dataset_patterns(args.datasets)
    if not dataset_paths:
        raise SystemExit("no dataset files were found")

    dataset = load_training_dataset(dataset_paths)
    FastLanguageModel, get_chat_template, TrainingArguments, SFTTrainer, torch = import_training_stack()
    model, tokenizer = load_model(args, FastLanguageModel, torch)
    dataset, tokenizer = apply_chat_template(dataset, tokenizer, get_chat_template)

    training_args = build_training_args(args, TrainingArguments, torch)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    resume_checkpoint = find_resume_checkpoint(Path(args.output)) if args.resume else None
    if resume_checkpoint:
        print(f"[resume] using checkpoint: {resume_checkpoint}")

    print_section("Training")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    save_adapter(Path(args.output), model, tokenizer)
    push_training_output(args, model, tokenizer, torch)
    print_summary(args, trainer_stats, len(dataset))


if __name__ == "__main__":
    main()
