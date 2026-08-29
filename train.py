import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer # INI UDAH PALING BARU

# 1. LOAD DATA
df = pd.read_csv("data_mtk.csv")
dataset = Dataset.from_pandas(df)

# 2. LOAD MODEL QWEN 0.5B
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu") # UDAH GANTI dtype

# 3. FUNGSI BUAT FORMAT DATA -> INI PENGGANTI dataset_text_field
def formatting_func(example):
    return f"Pertanyaan: {example['input']}\nJawaban: {example['output']}"

# 4. SETTING TRAINING
training_args = TrainingArguments(
    output_dir="./dewa_mtk_300mb",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=500,
    logging_steps=10,
)

# 5. JALANIN TRAINER VERSI BARU
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func, # GANTI INI
    max_seq_length=512,
    args=training_args,
)

print("MULAI TRAINING...")
trainer.train()
print("SELESAI! SAVE MODEL...")
trainer.save_model("./dewa_mtk_300mb")
