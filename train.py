import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

print("1. LOAD rumus.json...")
with open("rumus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
print(f"Data loaded: {len(dataset)} baris")

print("2. LOAD MODEL QWEN...")
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")

# INI KUNCINYA: UBAH "rumus" JADI FORMAT CHAT
def formatting_func(examples):
    texts = []
    for i in range(len(examples["rumus"])):
        rumus = examples["rumus"][i]
        kelas = examples["kelas"][i]
        
        # Bikin pertanyaan dari rumus nya
        if "=" in rumus and rumus.endswith("="):
            # Kalau "2 + 4 =" -> kita tanya "Berapa hasil dari 2 + 4?"
            pertanyaan = f"Kelas {kelas}: Berapa hasil dari {rumus.replace('=', '')}?"
            jawaban = f"Hasilnya adalah {eval(rumus.replace('=', ''))}" # auto hitung jawabannya
        else:
            # Kalau "4 + 2 = 6" -> kita tanya "Jelaskan rumus 4 + 2"
            pertanyaan = f"Kelas {kelas}: Jelaskan {rumus}"
            jawaban = f"Rumusnya adalah {rumus}"

        messages = [
            {"role": "user", "content": pertanyaan},
            {"role": "assistant", "content": jawaban}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts

print("3. MULAI TRAINING...")
training_args = TrainingArguments(
    output_dir="./dewa_mtk_300mb",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    max_sew_length=512,
    logging_steps=50,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
)

trainer.train()
trainer.save_model("./dewa_mtk_300mb")
print("4. SELESAI! MODEL DISIMPAN")
