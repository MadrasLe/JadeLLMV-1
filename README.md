# JadeLLM

![HF Downloads All Time](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/hf_total_all_time.json)
![HF Downloads 30d](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/hf_total_30d.json)
![Official Models](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/official_models.json)

JadeLLM is the public documentation and engineering repository for the Jade model family.

This repo exists to present the work clearly: the model lineup, the training direction, the maintained code paths, and the overall release structure behind Jade.

## What is in this repository

- [MODELS.md](./MODELS.md): public overview of the Jade family and its official variants.
- [PIPELINE.md](./PIPELINE.md): high-level explanation of the training and release workflow.
- [train_jade_sft.py](./train_jade_sft.py): maintained supervised fine-tuning entrypoint.
- [run_jade.py](./run_jade.py): simple local inference entrypoint for Transformers-compatible models.
- [models.json](./models.json): structured registry used to keep the official lineup consistent.

## Jade family

| Model | Params | Architecture | Base model | Format |
| --- | ---: | --- | --- | --- |
| [Jade-20B](https://huggingface.co/Madras1/Jade-20B) | 20.9B | `gpt_oss` | `openai/gpt-oss-20b` | Transformers |
| [Jade-14B](https://huggingface.co/Madras1/Jade-14B) | 14.8B | `qwen3` | `Qwen/Qwen3-14B` | Transformers |
| [Jade72b](https://huggingface.co/Madras1/Jade72b) | 72.7B | `qwen2` | `Qwen/Qwen2.5-72B` | Transformers |
| [Jade8b](https://huggingface.co/Madras1/Jade8b) | 8.2B | `qwen3` | not pinned in metadata | Transformers |
| [Jade4b](https://huggingface.co/Madras1/Jade4b) | 4.0B | `qwen3` | not pinned in metadata | Transformers |
| [Jade1.7b](https://huggingface.co/Madras1/Jade1.7b) | 1.7B | `qwen3` | `Qwen/Qwen3-1.7B` | Transformers |
| [Jade0.6b](https://huggingface.co/Madras1/Jade0.6b) | 0.6B | `qwen3` | not pinned in metadata | Transformers |
| [Jade8b-GGUF](https://huggingface.co/Madras1/Jade8b-GGUF) | 8B | GGUF export | Jade8b | GGUF |

The full narrative view of the lineup is in [MODELS.md](./MODELS.md).

## Training

The maintained training script is [train_jade_sft.py](./train_jade_sft.py).

It supports:

- chat datasets with `messages`
- Alpaca-style datasets with `instruction` and `output`
- simple Q&A datasets with `question` and `answer`

Example:

```bash
python train_jade_sft.py \
  --datasets path/to/dataset.jsonl \
  --model unsloth/Qwen2.5-3B \
  --adapter lora \
  --epochs 3 \
  --output outputs/jade-3b-lora
```

## Local inference

```bash
python run_jade.py \
  --repo Madras1/Jade-20B \
  --prompt "Jade, explica de forma simples o que eh LoRA"
```

## Project notes

- This repository is meant to document the work, not expose every internal scratch script.
- Public badges at the top summarize the official Jade family footprint directly on the GitHub page.
- Add a license file before treating the repo as a final public release.
