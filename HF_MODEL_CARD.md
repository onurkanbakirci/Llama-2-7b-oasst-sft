---
language:
- en
license: llama2
tags:
- llama-2
- conversational
- assistant
- sft
- lora
- qlora
base_model: meta-llama/Llama-2-7b-hf
datasets:
- OpenAssistant/oasst1
---

# Llama-2-7b Fine-tuned on OpenAssistant

This is a fine-tuned version of [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) trained on the [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset using supervised fine-tuning (SFT) with LoRA.

## Model Description

- **Base Model**: Llama-2-7b
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Training Dataset**: OpenAssistant Conversations Dataset (OASST1)
- **Task**: Conversational AI / Instruction Following
- **Language**: English

## Intended Uses

This model is designed for conversational AI applications where helpful, informative responses are needed. It can be used for:

- Chatbots and virtual assistants
- Question answering systems
- Interactive dialogue systems
- Educational tools

### Direct Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model and tokenizer
model_name = "onurkanbkrc/Llama-2-7b-oasst-sft"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Format prompt
prompt = """### User:
What is the capital of France?

### Assistant:
"""

# Generate response
output = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(output[0]["generated_text"])
```

## Training Details

### Training Data

The model was trained on the OpenAssistant Conversations Dataset (OASST1), which contains:
- High-quality human-generated conversations
- Multi-turn dialogues
- Assistant-style responses
- Diverse topics and question types

Conversations were formatted as:
```
### User:
[user message]

### Assistant:
[assistant response]
```

### Training Procedure

**Hardware**: Google Colab GPU (A100/T4)

**Training Hyperparameters**:
- **Epochs**: 2
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **LR Scheduler**: Cosine with 5% warmup
- **Optimizer**: Paged AdamW 8-bit
- **Max Sequence Length**: 1024 tokens
- **Precision**: BFloat16
- **Gradient Checkpointing**: Enabled
- **Sequence Packing**: Enabled

**LoRA Configuration**:
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dropout**: 0.05
- **Task Type**: Causal Language Modeling

**Quantization**:
- **Method**: 4-bit NF4 quantization (QLoRA)
- **Compute dtype**: BFloat16

### Training Infrastructure

- **Framework**: Hugging Face Transformers + PEFT + TRL
- **Monitoring**: LangSmith for real-time metrics
- **Memory Optimization**: 4-bit quantization, gradient checkpointing, paged optimizer

## Evaluation

Qualitative evaluation shows the fine-tuned model produces more helpful, contextual, and assistant-like responses compared to the base Llama-2-7b model.

## Limitations and Bias

- **Training Data Bias**: The model inherits biases present in the OpenAssistant dataset
- **Language**: Primarily trained on English conversations
- **Context Window**: Limited to 1024 tokens during training
- **Hallucination**: Like all LLMs, may generate plausible-sounding but incorrect information
- **Safety**: Not specifically fine-tuned for safety; may generate inappropriate content
- **Base Model Limitations**: Inherits limitations from Llama-2-7b

## Ethical Considerations

- This model should not be used for generating harmful, biased, or misleading content
- Users should implement appropriate safety measures and content filtering
- Model outputs should be reviewed, especially in high-stakes applications
- Follow Meta's Llama-2 Acceptable Use Policy

## Citation

If you use this model, please cite:

```bibtex
@misc{llama2-oasst-sft,
  author = {Onur Kanbakirci},
  title = {Llama-2-7b Fine-tuned on OpenAssistant},
  year = {2026},
  publisher = {HuggingFace Hub},
  url = {https://huggingface.co/onurkanbkrc/Llama-2-7b-oasst-sft}
}
```

### Base Model Citation

```bibtex
@article{touvron2023llama2,
  title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author={Hugo Touvron and Louis Martin and Kevin Stone and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

### QLoRA Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

## License

This model is subject to the [Llama 2 Community License Agreement](https://ai.meta.com/llama/license/). 

The training code and methodology are provided for educational purposes.

## Acknowledgments

- Meta AI for Llama-2
- OpenAssistant community for the dataset
- Hugging Face for the training infrastructure
- Tim Dettmers for QLoRA and bitsandbytes

## Model Card Authors

Onur Kanbakirci

## Model Card Contact

For questions or issues, please open an issue on the model repository.
