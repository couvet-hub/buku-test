import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

print("CUDA:", torch.cuda.is_available())
model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("json", data_files="rumus.json")

training_args = TrainingArguments(
    output_dir="./dewa_mtk_300mb",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    fp16=True
)

trainer = SFTTrainer(model=model, train_dataset=dataset["train"], dataset_text_field="output", max_seq_length=512, args=training_args)
trainer.train()
trainer.save_model("./dewa_mtk_300mb")
print("SELESAI TRAINING")
