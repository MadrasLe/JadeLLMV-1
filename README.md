# JadeLLM

![HF Downloads All Time](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/hf_total_all_time.json)
![HF Downloads 30d](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/hf_total_30d.json)
![Official Models](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/official_models.json)
![Community Downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MadrasLe/JadeLLMV-1/main/badges/hf_community_all_time.json)

**JadeLLM** é o repositório público de engenharia e documentação da família Jade — uma linha de modelos de linguagem treinados com foco em **português brasileiro**, persona consistente e tom conversacional natural.

Jade fala como gente. Não como manual.

---

## O que é a Jade

Jade é uma família de LLMs fine-tunados para chat em PT-BR com persona forte e integrada aos pesos do modelo. Isso significa que ela mantém um estilo próprio — informal, direto, humano — mesmo sem um system prompt elaborado.

O estilo inclui:
- linguagem coloquial e abreviações naturais (`vc`, `tlgd`, `tmj`)
- tom de conversa, não de documentação
- respostas curtas quando faz sentido, desenvolvidas quando necessário
- personalidade consistente independente do tamanho do modelo

A família vai de **0.6B até 72B de parâmetros**, com diferentes arquiteturas base e casos de uso distintos.

---

## Família de modelos

| Modelo | Params | Base | Arquitetura | Formato |
| --- | ---: | --- | --- | --- |
| [Jade-20B](https://huggingface.co/Madras1/Jade-20B) | 20.9B | `openai/gpt-oss-20b` | `gpt_oss` | Transformers |
| [Jade-14B](https://huggingface.co/Madras1/Jade-14B) | 14.8B | `Qwen/Qwen3-14B` | `qwen3` | Transformers |
| [Jade72b](https://huggingface.co/Madras1/Jade72b) | 72.7B | `Qwen/Qwen2.5-72B` | `qwen2` | Transformers |
| [Jade8b](https://huggingface.co/Madras1/Jade8b) | 8.2B | Qwen3 | `qwen3` | Transformers |
| [Jade4b](https://huggingface.co/Madras1/Jade4b) | 4.0B | Qwen3 | `qwen3` | Transformers |
| [Jade1.7b](https://huggingface.co/Madras1/Jade1.7b) | 1.7B | `Qwen/Qwen3-1.7B` | `qwen3` | Transformers |
| [Jade0.6b](https://huggingface.co/Madras1/Jade0.6b) | 0.6B | Qwen3 | `qwen3` | Transformers |
| [Jade8b-GGUF](https://huggingface.co/Madras1/Jade8b-GGUF) | 8B | Jade8b | GGUF export | GGUF |

A coleção completa está em [huggingface.co/collections/Madras1/jade-v1](https://huggingface.co/collections/Madras1/jade-v1).

---

## Inferência local

```bash
python run_jade.py \
  --repo Madras1/Jade-20B \
  --prompt "jade, me explica o que é LoRA de um jeito simples"
```

Ou direto com Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Madras1/Jade-20B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "oi jade, tudo bem?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Para rodar localmente com llama.cpp ou compatíveis, use [Jade8b-GGUF](https://huggingface.co/Madras1/Jade8b-GGUF).

---

## Treinamento

O script mantido publicamente é [train_jade_sft.py](./train_jade_sft.py).

Ele aceita três formatos de dataset JSONL:

| Formato | Campos esperados |
| --- | --- |
| Chat | `messages` (lista de turnos com `role` e `content`) |
| Alpaca | `instruction`, `input` (opcional), `output` |
| Q&A simples | `question` / `prompt` + `answer` / `response` |

Suporta LoRA, DoRA e full fine-tuning via [Unsloth](https://github.com/unslothai/unsloth).

```bash
python train_jade_sft.py \
  --datasets path/to/dataset.jsonl \
  --model unsloth/Qwen2.5-3B \
  --adapter lora \
  --epochs 3 \
  --output outputs/jade-3b-lora
```

Depois do treino, para fazer merge e publicar:

```bash
python train_jade_sft.py \
  --datasets data.jsonl \
  --model unsloth/Qwen2.5-3B \
  --adapter lora \
  --save-merged \
  --push-to-hub SeuUser/seu-modelo
```

A documentação completa do pipeline está em [PIPELINE.md](./PIPELINE.md).

---

## Estrutura do repositório

```
JadeLLM/
├── train_jade_sft.py          # entrypoint de treinamento SFT
├── run_jade.py                # inferência local
├── sync_hf_metrics.py         # sincroniza métricas do HF para os badges
├── models.json                # registro estruturado da família
├── MODELS.md                  # visão narrativa da linha de modelos
├── PIPELINE.md                # explicação do fluxo de treino e release
├── badges/                    # JSONs para os badges dinâmicos do README
├── hf_metrics_snapshot.json   # snapshot mais recente de downloads
└── hf_metrics_history.json    # histórico de métricas ao longo do tempo
```

---
Comparação antes/após SFT;

<img width="934" height="613" alt="jade2" src="https://github.com/user-attachments/assets/99c1f293-0120-4468-8ee1-9e71ee190006" />

<img width="922" height="734" alt="jade" src="https://github.com/user-attachments/assets/68e583b5-5c4b-475e-ac43-ebe0517553af" />
