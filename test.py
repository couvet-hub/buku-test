from transformers import pipeline
import torch

dewa = pipeline("text-generation", model="./dewa_mtk_300mb", dtype=torch.float16, device_map="cpu")

pertanyaan = "Kelas 1: Berapa hasil dari 2 + 4?"
messages = [{"role": "user", "content": pertanyaan}]
prompt = dewa.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

hasil = dewa(prompt, max_new_tokens=100)
print(hasil[0]['generated_text'])
