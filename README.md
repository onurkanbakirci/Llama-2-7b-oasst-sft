# Base to SFT

A minimal code example for Supervised Fine-Tuning (SFT) to convert a base language model into a conversational chat model.

## Overview

`sft.py` demonstrates how to fine-tune Meta's Llama-2-7b base model on the OpenAssistant dataset to create a helpful chat assistant using parameter-efficient training techniques.

## Training Pipeline

```mermaid
graph TD
    A[Pre-Training Phase] -->|Web-scale text| B[Base Model]
    B -->|General language understanding| C[Llama-2-7b Base]
    
    C -->|SFT Phase| D[Supervised Fine-Tuning]
    D -->|Conversational dataset| E[Chat Model]
    
    F[OpenAssistant Dataset] -->|User-Assistant pairs| D
    
    D --> G[Apply LoRA]
    G --> H[4-bit Quantization]
    H --> I[Train on conversations]
    I --> E
    
    E -->|Post-Training| J[Evaluation]
    J --> K[Deployment]
    
    style A fill:#e1f5ff
    style D fill:#fff4e1
    style J fill:#f0e1ff
    style C fill:#ff9999
    style E fill:#99ff99
```

## Model Evolution

### Base vs Chat Model Comparison

![Prompt Evaluation](assets/prompt-eval.png)

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets peft trl bitsandbytes

# Run training
python sft.py
```

## What's Inside

- **Model Loading**: 4-bit quantized Llama-2-7b
- **LoRA**: Parameter-efficient fine-tuning
- **Dataset**: OpenAssistant conversational data
- **Training**: 2 epochs with cosine scheduler
- **Output**: Fine-tuned chat model

### GPU Usage During Training

![GPU Usage](assets/gpu-usage.png)
