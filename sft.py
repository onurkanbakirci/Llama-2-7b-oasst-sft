# =========================
# Imports
# =========================

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from langsmith import Client as LangSmithClient
from transformers import TrainerCallback
import uuid
import datetime


# =========================
# Config
# =========================

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "OpenAssistant/oasst1"

OUTPUT_DIR = "./sft_llama_oasst"

MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACC = 4
EPOCHS = 2
LR = 2e-4


# =========================
# LangSmith Setup
# =========================

from google.colab import userdata

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = userdata.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "SFT-OASST1"


# =========================
# LangSmith Trainer Callback
# =========================

class LangSmithTrainerCallback(TrainerCallback):
    """
    Custom HuggingFace TrainerCallback that logs training metrics
    to LangSmith so you can monitor runs on the dashboard.
    """

    def __init__(self, project_name=None):
        self.client = LangSmithClient()
        self.project_name = project_name or os.environ.get("LANGSMITH_PROJECT", "SFT-Training")
        self.parent_run_id = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.parent_run_id = uuid.uuid4()
        self.client.create_run(
            name="SFT Training",
            run_type="chain",
            id=self.parent_run_id,
            project_name=self.project_name,
            inputs={
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "epochs": args.num_train_epochs,
                "batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "max_length": MAX_SEQ_LENGTH,
                "grad_accumulation": args.gradient_accumulation_steps,
            },
            start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        print("LangSmith: training run created.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.parent_run_id is None:
            return

        step = state.global_step
        child_run_id = uuid.uuid4()

        self.client.create_run(
            name=f"step-{step}",
            run_type="chain",
            id=child_run_id,
            parent_run_id=self.parent_run_id,
            project_name=self.project_name,
            inputs={"global_step": step},
            start_time=datetime.datetime.now(datetime.timezone.utc),
        )

        self.client.update_run(
            run_id=child_run_id,
            outputs={k: v for k, v in logs.items()},
            end_time=datetime.datetime.now(datetime.timezone.utc),
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.parent_run_id is None:
            return

        final_metrics = {}
        if state.log_history:
            final_metrics = {k: v for k, v in state.log_history[-1].items()}
        final_metrics["total_steps"] = state.global_step

        self.client.update_run(
            run_id=self.parent_run_id,
            outputs=final_metrics,
            end_time=datetime.datetime.now(datetime.timezone.utc),
        )
        print("LangSmith: training run completed.")


# =========================
# Load Tokenizer & Model
# =========================

print("Loading tokenizer & model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False


# =========================
# LoRA Setup
# =========================

print("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# Load Dataset
# =========================

print("Loading OpenAssistant/oasst1 dataset...")

dataset = load_dataset(DATASET_NAME)


# =========================
# Build Conversation Lookup
# =========================

print("Building lookup map...")

id_to_text = {}

for row in dataset["train"]:
    id_to_text[row["message_id"]] = row["text"]


# =========================
# Format Dataset (Prompt → Answer)
# =========================

print("Formatting examples...")

def format_example(example):

    if example["role"] != "assistant":
        return None

    parent_id = example["parent_id"]

    if parent_id not in id_to_text:
        return None

    user_text = id_to_text[parent_id]
    assistant_text = example["text"]

    text = f"""### User:
{user_text}

### Assistant:
{assistant_text}
"""

    return {"text": text}


formatted_data = []

for ex in dataset["train"]:
    item = format_example(ex)
    if item:
        formatted_data.append(item)


print("Final SFT samples:", len(formatted_data))


from datasets import Dataset
sft_dataset = Dataset.from_list(formatted_data)


# =========================
# Training Arguments
# =========================

print("Training config...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,

    learning_rate=LR,
    num_train_epochs=EPOCHS,

    fp16=False,
    bf16=True,

    logging_steps=20,
    save_steps=500,
    save_total_limit=2,

    eval_strategy="no",

    report_to="none",

    optim="paged_adamw_8bit",

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    gradient_checkpointing=True,

    max_length=MAX_SEQ_LENGTH,
    packing=True,
    dataset_text_field="text",
)


# =========================
# SFT Trainer
# =========================

print("Initializing SFTTrainer...")


langsmith_callback = LangSmithTrainerCallback()

trainer = SFTTrainer(
    model=model,

    train_dataset=sft_dataset,

    args=training_args,

    processing_class=tokenizer,

    callbacks=[langsmith_callback],
)


# =========================
# Train
# =========================

print("Starting training...")

trainer.train()

print("Saving model...")

trainer.save_model(OUTPUT_DIR)

print("Training complete!")
