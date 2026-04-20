# Jade Models

Documentação narrativa da linha oficial de modelos da família Jade.
O registro estruturado com metadados vive em [models.json](./models.json).

---

## Linha oficial

| Modelo | Repo | Params | Base | Arquitetura | Formato |
| --- | --- | ---: | --- | --- | --- |
| Jade-20B | [Madras1/Jade-20B](https://huggingface.co/Madras1/Jade-20B) | 20.9B | `openai/gpt-oss-20b` | `gpt_oss` | Transformers |
| Jade-14B | [Madras1/Jade-14B](https://huggingface.co/Madras1/Jade-14B) | 14.8B | `Qwen/Qwen3-14B` | `qwen3` | Transformers |
| Jade72b | [Madras1/Jade72b](https://huggingface.co/Madras1/Jade72b) | 72.7B | `Qwen/Qwen2.5-72B` | `qwen2` | Transformers |
| Jade8b | [Madras1/Jade8b](https://huggingface.co/Madras1/Jade8b) | 8.2B | Qwen3 | `qwen3` | Transformers |
| Jade4b | [Madras1/Jade4b](https://huggingface.co/Madras1/Jade4b) | 4.0B | Qwen3 | `qwen3` | Transformers |
| Jade1.7b | [Madras1/Jade1.7b](https://huggingface.co/Madras1/Jade1.7b) | 1.7B | `Qwen/Qwen3-1.7B` | `qwen3` | Transformers |
| Jade0.6b | [Madras1/Jade0.6b](https://huggingface.co/Madras1/Jade0.6b) | 0.6B | Qwen3 | `qwen3` | Transformers |
| Jade8b-GGUF | [Madras1/Jade8b-GGUF](https://huggingface.co/Madras1/Jade8b-GGUF) | 8B | Jade8b | GGUF export | GGUF |

---

## Como ler a linha

A família cobre um espectro deliberado de tamanhos e casos de uso.

**Modelos de cabeça de linha**

`Jade-20B` e `Jade72b` são os modelos de maior capacidade da família. O Jade72b foi o primeiro release da família, construído sobre Qwen2.5-72B. O Jade-20B é o mais recente dos grandes, usando `openai/gpt-oss-20b` como base — uma das primeiras tentativas de fine-tune público desse modelo em português.

**Faixa intermediária**

`Jade-14B` e `Jade8b` equilibram qualidade e custo de inferência. São os mais adequados para quem quer um modelo capaz rodando em hardware razoável.

**Modelos compactos**

`Jade4b`, `Jade1.7b` e `Jade0.6b` são os menores da família. Úteis para hardware limitado, integração em apps móveis, ou cenários onde o custo de memória e latência importa mais que a capacidade máxima. O Jade1.7b em particular tem uma relação qualidade/tamanho surpreendente para PT-BR.

**Formato alternativo**

`Jade8b-GGUF` é o export do Jade8b em formato GGUF para rodar com llama.cpp e ferramentas compatíveis como Ollama e LM Studio. É a porta de entrada para quem quer rodar Jade localmente sem depender de Python ou CUDA.

---

## Comportamento esperado

Todos os modelos da família Jade são treinados com o mesmo objetivo central: **a persona vive nos pesos, não só no prompt**.

Na prática, isso significa que qualquer modelo Jade vai:
- responder em PT-BR por padrão
- usar linguagem informal e coloquial quando o prompt pedir isso
- manter um tom consistente mesmo ao longo de conversas longas

O grau de consistência e qualidade escala com o tamanho do modelo, mas o caráter base é o mesmo.

---

## Por que manter `models.json`

O `models.json` é o source of truth estruturado da linha. Ele alimenta os badges automáticos do repositório e serve como referência centralizada para nomes, repos e metadados de cada modelo.

A narrativa está aqui. Os dados ficam lá.
