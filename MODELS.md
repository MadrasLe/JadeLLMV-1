# Jade Models

This file is the narrative companion to [models.json](./models.json).

## Official family

| Model | Repo | Params | Architecture | Base model | Format |
| --- | --- | ---: | --- | --- | --- |
| Jade-20B | `Madras1/Jade-20B` | 20.9B | `gpt_oss` | `openai/gpt-oss-20b` | Transformers |
| Jade-14B | `Madras1/Jade-14B` | 14.8B | `qwen3` | `Qwen/Qwen3-14B` | Transformers |
| Jade72b | `Madras1/Jade72b` | 72.7B | `qwen2` | `Qwen/Qwen2.5-72B` | Transformers |
| Jade8b | `Madras1/Jade8b` | 8.2B | `qwen3` | not pinned in metadata | Transformers |
| Jade4b | `Madras1/Jade4b` | 4.0B | `qwen3` | not pinned in metadata | Transformers |
| Jade1.7b | `Madras1/Jade1.7b` | 1.7B | `qwen3` | `Qwen/Qwen3-1.7B` | Transformers |
| Jade0.6b | `Madras1/Jade0.6b` | 0.6B | `qwen3` | not pinned in metadata | Transformers |
| Jade8b-GGUF | `Madras1/Jade8b-GGUF` | 8B | exported variant | Jade8b | GGUF |

## Reading the lineup

One useful way to interpret the family is:

- `Jade72b` and `Jade-20B` as larger headline releases,
- `Jade-14B` and `Jade8b` as more balanced mid-range variants,
- `Jade4b`, `Jade1.7b`, and `Jade0.6b` as smaller deployment-oriented variants,
- `Jade8b-GGUF` as the local / llama.cpp-style export path.

## Why keep `models.json`

The public-facing explanation is in this file, but the structured model list lives in `models.json` so the repository keeps one consistent source of truth for names, links, and metadata.
