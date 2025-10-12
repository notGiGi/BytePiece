#!/usr/bin/env python3
from bytepiece import train_bpe
from bytepiece.visualization import show_tokens, TokenStats

corpus = [
    "def calculate(): return x >= 10",
    "if status >= 200: return True",
    'url = "https://api.example.com/users"',
    "for i in range(100): print(i)",
]

print("BPE...")
encoder = train_bpe(
    texts=corpus,
    vocab_size=200
)

print("\n" + "="*70)
print("VISUALIZACIÓN 1: Inline")
print("="*70)
show_tokens(encoder, "def calculate(): return x >= 10", style="inline")

print("\n" + "="*70)
print("VISUALIZACIÓN 2: Tabla")
print("="*70)
show_tokens(encoder, "def calculate(): return x >= 10", style="table")

print("\n" + "="*70)
print("VISUALIZACIÓN 3: Coloreado")
print("="*70)
show_tokens(encoder, "def calculate(): return x >= 10", style="colored")


print("\n" + "="*70)
print("ESTADÍSTICAS DEL CORPUS")
print("="*70)
all_tokens = []
for text in corpus:
    tokens = encoder.encode(text)
    all_tokens.extend([encoder.vocab.id_to_token.get(tid, f"<{tid}>") 
                       for tid in tokens])

stats = TokenStats()
stats.show_stats(all_tokens)