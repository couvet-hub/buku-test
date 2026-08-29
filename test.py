from transformers import pipeline
import torch

print("LOADING MODEL 300MB DARI FOLDER...")
dewa = pipeline("text-generation", model="./dewa_mtk_300mb", torch_dtype=torch.float16, device_map="cpu", max_new_tokens=150)

pertanyaan = [
    "Jelaskan rumus luas persegi untuk kelas 3 SD",
    "Apa rumus keliling lingkaran kelas 6? Beri contoh"
]

for i, q in enumerate(pertanyaan):
    print(f"\n=== TES {i+1}: {q} ===")
    hasil = dewa(q)
    print(hasil[0]['generated_text'])

print("\nSELESAI TES. CEK JAWABAN DI ATAS")
