# Base to SFT: Llama-2 Fine-Tuning on OpenAssistant

A supervised fine-tuning (SFT) pipeline for adapting Llama-2-7b into a conversational assistant using the OpenAssistant dataset. This project demonstrates parameter-efficient fine-tuning with LoRA, training monitoring with LangSmith, and model deployment to Hugging Face Hub.

## 🎯 Overview

This project fine-tunes Meta's Llama-2-7b model on the OpenAssistant/oasst1 dataset to create a helpful conversational AI assistant. The training uses:

- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **4-bit quantization** to reduce memory requirements
- **LangSmith integration** for real-time training monitoring
- **QLoRA approach** combining quantization and LoRA

## ✨ Features

- 🚀 **Efficient Training**: Uses LoRA with 4-bit quantization to train on consumer GPUs
- 📊 **Training Monitoring**: LangSmith callback tracks metrics in real-time
- 🔄 **Conversation Formatting**: Automatically builds prompt-response pairs from OpenAssistant data
- 📈 **Model Comparison**: Built-in inference comparison between base and fine-tuned models
- 🤗 **Easy Deployment**: Push trained models directly to Hugging Face Hub
- 🎓 **Best Practices**: Implements gradient checkpointing, mixed precision, and optimal scheduler

## 📋 Requirements

### Python Dependencies

```bash
pip install torch transformers datasets
pip install peft trl bitsandbytes
pip install langsmith
pip install huggingface_hub
```

### Environment Setup

This project is designed to run on Google Colab but can be adapted for local use. You'll need:

- **GPU**: CUDA-compatible GPU with at least 16GB VRAM (for 4-bit training)
- **Storage**: ~15GB for model checkpoints
- **API Keys**:
  - LangSmith API key (for training monitoring)
  - Hugging Face token (for model download and upload)
  - Meta Llama-2 access (request access on Hugging Face)

## 🚀 Usage

### 1. Training (`sft.py`)

Fine-tune Llama-2-7b on the OpenAssistant dataset:

```bash
python sft.py
```

**What it does:**
- Loads Llama-2-7b with 4-bit quantization
- Applies LoRA adapters to attention and MLP layers
- Formats OpenAssistant conversations into User/Assistant format
- Trains for 2 epochs with cosine learning rate schedule
- Saves checkpoints to `./sft_llama_oasst`
- Logs all metrics to LangSmith

**Key Configuration:**
```python
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "OpenAssistant/oasst1"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACC = 4
EPOCHS = 2
LR = 2e-4
```

### 2. Model Comparison (`sft-inference.py`)

Compare base model vs fine-tuned model outputs:

```bash
python sft-inference.py
```

**Output Example:**
```
+------------------+------------------------------------------+------------------------------------------+
|                  | Base Model                               | Chat Model (SFT)                         |
+------------------+------------------------------------------+------------------------------------------+
| Model Name       | meta-llama/Llama-2-7b-hf                | onurkanbakirci/Llama-2-7b-oasst-sft     |
+------------------+------------------------------------------+------------------------------------------+
| Prompt           | Are you a helpful assistant?             | Are you a helpful assistant?             |
+------------------+------------------------------------------+------------------------------------------+
| Response         | [unformatted continuation]               | Yes! I'm here to help you with...        |
+------------------+------------------------------------------+------------------------------------------+
```

### 3. Push to Hugging Face (`sft-push-to-hf.py`)

Deploy your fine-tuned model to Hugging Face Hub:

```bash
python sft-push-to-hf.py
```

**What it does:**
- Merges LoRA adapters into the base model
- Uploads merged model to Hugging Face Hub
- Uploads tokenizer configuration
- Makes model publicly accessible

## 🏗️ Project Structure

```
base-to-sft/
├── sft.py                   # Main training script
├── sft-inference.py         # Model comparison tool
├── sft-push-to-hf.py       # HuggingFace deployment script
└── sft_llama_oasst/        # Output directory (created during training)
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── ...
```

## 🔧 Configuration Details

### LoRA Configuration

```python
LoraConfig(
    r=16,                    # Rank of LoRA matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Attention and MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Training Configuration

- **Optimizer**: Paged AdamW 8-bit
- **Precision**: BFloat16
- **Learning Rate**: 2e-4 with cosine schedule
- **Warmup**: 5% of total steps
- **Gradient Checkpointing**: Enabled
- **Sequence Packing**: Enabled for efficiency

### Quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

## 📊 Dataset Format

The training script converts OpenAssistant conversations into this format:

```
### User:
[User's question or prompt]

### Assistant:
[Assistant's response]
```

Only assistant messages with valid parent prompts are included, ensuring high-quality training pairs.

## 🎓 Training Tips

1. **Memory Issues?**
   - Reduce `BATCH_SIZE` or increase `GRAD_ACC`
   - Decrease `MAX_SEQ_LENGTH`
   - Use 8-bit quantization instead of 4-bit

2. **Training Time**
   - ~2-4 hours on A100 GPU
   - ~4-8 hours on T4 GPU
   - Adjust `EPOCHS` based on convergence

3. **Model Quality**
   - Monitor loss curves in LangSmith
   - Test with diverse prompts during training
   - Consider validation split for better evaluation

## 🔑 Environment Variables

For Google Colab, set these in Colab Secrets:

```python
LANGSMITH_API_KEY      # LangSmith monitoring
HF_WRITE_TOKEN         # Hugging Face upload
```

For local use, export as environment variables:

```bash
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_PROJECT="SFT-OASST1"
export HF_TOKEN="your-token"
```

## 📈 Monitoring with LangSmith

The project includes a custom `LangSmithTrainerCallback` that logs:

- Training hyperparameters
- Loss and learning rate per step
- Gradient norms
- Training duration
- Final model metrics

Access your dashboard at: https://smith.langchain.com/

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Add evaluation metrics (perplexity, BLEU)
- Support for other base models
- Multi-GPU training
- Validation set integration
- More dataset options

## 📝 License

This project is for educational purposes. Please respect:

- **Llama-2 License**: Follow Meta's acceptable use policy
- **OpenAssistant Dataset**: Apache 2.0 License
- **Dependencies**: Check individual package licenses

## 🙏 Acknowledgments

- **Meta AI** for Llama-2
- **OpenAssistant** for the conversation dataset
- **Hugging Face** for transformers, PEFT, and TRL libraries
- **LangSmith** for training monitoring tools
- **Tim Dettmers** for QLoRA and bitsandbytes

## 📚 Resources

- [Llama-2 Paper](https://arxiv.org/abs/2307.09288)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [OpenAssistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Hugging Face SFT Documentation](https://huggingface.co/docs/trl/sft_trainer)

---

**Author**: Onur Kanbakirci  
**Model**: [onurkanbakirci/Llama-2-7b-oasst-sft](https://huggingface.co/onurkanbkrc/Llama-2-7b-oasst-sft)
