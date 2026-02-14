# =========================
# Base Model vs SFT Model Comparison
# =========================

from transformers import AutoModelForCausalLM, pipeline
import torch

PROMPT = "Are you a helpful assistant?"
FORMATTED_PROMPT = f"### User:\n{PROMPT}\n\n### Assistant:\n"
MAX_NEW_TOKENS = 30


# --- Load Base Model ---

print("Loading base model for comparison...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

base_pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    device_map="auto",
)


# --- Load SFT Model ---

print("Loading SFT model...")

sft_pipe = pipeline(
    "text-generation",
    model=OUTPUT_DIR,
    tokenizer=tokenizer,
    device_map="auto",
)


# --- Generate ---

def generate(pipe, prompt):
    out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7)
    return out[0]["generated_text"][len(prompt):].strip()


base_output = generate(base_pipe, PROMPT)
sft_output = generate(sft_pipe, FORMATTED_PROMPT)


# --- Print Table ---

label_col = ""
col_base = "Base Model"
col_sft = "Chat Model (SFT)"
w0 = 16
w1 = 40
w2 = 40
sep = f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+"

def wrap(text, width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line = f"{line} {word}" if line else word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]

def print_row(label, text1, text2):
    lines1 = wrap(text1, w1)
    lines2 = wrap(text2, w2)
    max_lines = max(len(lines1), len(lines2))
    lines1 += [""] * (max_lines - len(lines1))
    lines2 += [""] * (max_lines - len(lines2))
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        lbl = label if i == 0 else ""
        print(f"| {lbl:<{w0}} | {l1:<{w1}} | {l2:<{w2}} |")

print()
print(sep)
print(f"| {label_col:<{w0}} | {col_base:<{w1}} | {col_sft:<{w2}} |")
print(sep)
SFT_MODEL_NAME = "onurkanbakirci/Llama-2-7b-oasst-sft"
print_row("Model Name", MODEL_NAME, SFT_MODEL_NAME)
print(sep)
print_row("Prompt", PROMPT, PROMPT)
print(sep)
print_row("Response", base_output, sft_output)
print(sep)
