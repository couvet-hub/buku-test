import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 1. LOAD DATA DARI JSON KAMU LANGSUNG
with open("rumus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data) # dataset sekarang punya kolom 'input' dan 'output'

# 2. LOAD MODEL + TOKENIZER QWEN
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")

# 3. INI KUNCINYA: TERJEMAHIN 'input' 'output' JADI FORMAT CHAT QWEN
def formatting_func(example):
    messages = [
        {"role": "user", "content": example["input"]}, # ambil dari kolom 'input'
        {"role": "assistant", "content": example["output"]} # ambil dari kolom 'output'
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# 4. SETTING TRAINING
training_args = TrainingArguments(
    output_dir="./dewa_mtk_300mb",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_strategy="no" # biar ga save tengah2, langsung akhir aja
)

# 5. JALANIN TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func, # dia bakal baca 'input' 'output' dari sini
    max_seq_length=512,
    args=training_args,
)

print("MULAI TRAINING 894 RUMUS...")
trainer.train()
trainer.save_model("./dewa_mtk_300mb")
print("SELESAI! MODEL DISIMPAN")
