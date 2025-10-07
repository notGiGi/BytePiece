from bytepiece import train_bpe, BPEEncoder
from bytepiece.core.normalizer import Normalizer, PreTokenizationMode, SpacerMode

texts = [
    "Hello world!",
    "Hello there!",
    "I don't know",
    "I don't think so",
    "John's book",
    "John's friend",
    "They're going home",
    "They're coming back",
    "We're learning",
    "We're testing",
]
print("********************************************************")
print("Test 1: Training BPE WITHOUT pre-tokenization (NONE)")


normalizer_none = Normalizer(
    pre_tokenization=PreTokenizationMode.NONE,
    spacer_mode=SpacerMode.PREFIX,
)

vocab1, merges1, _ = train_bpe(
    texts=texts,
    vocab_size=500,  
    normalizer=normalizer_none,
    byte_fallback=True,
    verbose=True,
)

encoder1 = BPEEncoder(vocab1, merges1, normalizer_none)

print("\nTokenizing 'I don\\'t know':")
print("------------------------------------")
tokens1 = encoder1.encode("I don't know")
print(f"  Tokens: {tokens1}")
print(f"  Number of tokens: {len(tokens1)}")

print("=================================================")
print("Test 2: Training BPE WITH pre-tokenization (GPT2)")


normalizer_gpt2 = Normalizer(
    pre_tokenization=PreTokenizationMode.GPT2,
    spacer_mode=SpacerMode.PREFIX,
)

vocab2, merges2, _ = train_bpe(
    texts=texts,
    vocab_size=500, 
    normalizer=normalizer_gpt2,
    byte_fallback=True,
    verbose=True,
)

encoder2 = BPEEncoder(vocab2, merges2, normalizer_gpt2)

print("\nTokenizing 'I don\\'t know':")
tokens2 = encoder2.encode("I don't know")
print(f"  Tokens: {tokens2}")
print(f"  Number of tokens: {len(tokens2)}")

print("\n" + "=" * 70)
print("Comparison:")
print("=" * 70)
print(f"WITHOUT pre-tokenization: {len(tokens1)} tokens")
print(f"  {tokens1}")
print(f"\nWITH GPT-2 pre-tokenization: {len(tokens2)} tokens")
print(f"  {tokens2}")

print("\n" + "=" * 70)
print("Top 10 Merges Learned:")
print("=" * 70)

print("\nWITHOUT pre-tokenization:")
if len(merges1) > 0:
    for i, (left, right) in enumerate(merges1.merges[:10]):
        print(f"  {i+1}. '{left}' + '{right}' → '{left + right}'")
else:
    print("  (No merges learned)")

print("\nWITH GPT-2 pre-tokenization:")
if len(merges2) > 0:
    for i, (left, right) in enumerate(merges2.merges[:10]):
        print(f"  {i+1}. '{left}' + '{right}' → '{left + right}'")
else:
    print("  (No merges learned)")
