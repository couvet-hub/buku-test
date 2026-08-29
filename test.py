from transformers import pipeline
import torch

print("LOAD MODEL...")
dewa = pipeline("text-generation", model="./dewa_mtk_300mb", dtype=torch.float16, device_map="cpu")

pertanyaan = "Apa rumus luas persegi kelas 3 SD?"
messages = [{"role": "user", "content": pertanyaan}]
prompt = dewa.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("JAWABAN:")
hasil = dewa(prompt, max_new_tokens=150)
print(hasil[0]['generated_text'])
