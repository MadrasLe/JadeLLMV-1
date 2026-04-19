# Jade Pipeline

This file summarizes the work behind the Jade family at a high level.

The point here is not to expose every experiment. It is to explain the logic of the project in a way that makes the overall workflow understandable.

## 1. Data shaping

Before training, the dataset has to be shaped into a consistent conversational format.

That stage can involve:

- collecting or selecting source material,
- curating dialogue-style examples,
- generating or refining prompts,
- distilling target-style responses,
- filtering weak samples,
- and normalizing everything into a clean chat structure.

Those utilities are part of the broader history of the project, even when they are not all exposed as maintained public scripts.

## 2. Supervised fine-tuning

The maintained public training entrypoint in this repo is [train_jade_sft.py](./train_jade_sft.py).

That script handles the stable part of the workflow:

- loading JSONL datasets,
- normalizing different input formats,
- building a consistent chat view of the data,
- applying a template,
- and training with LoRA, DoRA, or full fine-tuning.

## 3. Release and usage

Once a model is trained, the focus shifts to release quality:

- packaging the result,
- optionally merging adapters,
- checking that inference behaves as expected,
- and making the final artifact usable through a clean public interface.

The maintained local inference entrypoint in this repo is [run_jade.py](./run_jade.py).

## Why this repo is selective

This repository is meant to document the work clearly, not to act as a dump of every temporary tool used along the way.

That is why the repo keeps the public-facing pieces:

- the model family view,
- the maintained training path,
- the maintained inference path,
- and the structured release registry.
