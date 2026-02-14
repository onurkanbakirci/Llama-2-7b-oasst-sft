# =========================
# Push SFT Model to Hugging Face Hub
# =========================

from google.colab import userdata
from huggingface_hub import login

# --- Config ---

HF_REPO_NAME = "onurkanbkrc/Llama-2-7b-oasst-sft"

# --- Login to Hugging Face ---

login(token=userdata.get("HF_WRITE_TOKEN"))

# --- Merge LoRA into base model ---

print("Merging LoRA weights into base model...")

merged_model = trainer.model.merge_and_unload()

# --- Push to Hub ---

print(f"Pushing merged model to {HF_REPO_NAME}...")

merged_model.push_to_hub(HF_REPO_NAME)
tokenizer.push_to_hub(HF_REPO_NAME)

print(f"Done! Model available at: https://huggingface.co/{HF_REPO_NAME}")