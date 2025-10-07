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

test_sentence = "I don't know"

print("*" * 10)
print("Training BPE WITHOUT pre-tokenization (NONE)")
print("*" * 10)

normalizer_none = Normalizer(
    pre_tokenization=PreTokenizationMode.NONE,
    spacer_mode=SpacerMode.PREFIX,
)

vocab1, merges1, _ = train_bpe(
    texts=texts,
    vocab_size=500,
    normalizer=normalizer_none,
    byte_fallback=True,
    verbose=False,  
)

encoder1 = BPEEncoder(vocab1, merges1, normalizer_none)

print("\n" + "=" * 70)
print("Training BPE WITH pre-tokenization (GPT2)")
print("=" * 70)

normalizer_gpt2 = Normalizer(
    pre_tokenization=PreTokenizationMode.GPT2,
    spacer_mode=SpacerMode.PREFIX,
)

vocab2, merges2, _ = train_bpe(
    texts=texts,
    vocab_size=500,
    normalizer=normalizer_gpt2,
    byte_fallback=True,
    verbose=False,
)

encoder2 = BPEEncoder(vocab2, merges2, normalizer_gpt2)


def decode_token(token):

    if '<0x' in token:
  
        import re
        byte_pattern = r'<0x([0-9A-F]{2})>'
        hex_values = re.findall(byte_pattern, token)
        
        if hex_values:
       
            byte_values = bytes(int(h, 16) for h in hex_values)
            try:
                return byte_values.decode('utf-8')
            except:
                return token  
    return token


print("\n" + "=" * 70)
print(f"Tokenizing: '{test_sentence}'")
print("=" * 70)


tokens1 = encoder1.encode(test_sentence)
decoded_tokens1 = [decode_token(t) for t in tokens1]

print("\nWITHOUT pre-tokenization:")
print(f"  Raw tokens ({len(tokens1)}): {tokens1}")
print(f"  Decoded: {decoded_tokens1}")
print(f"  Reconstructed: {''.join(decoded_tokens1)}")


tokens2 = encoder2.encode(test_sentence)
decoded_tokens2 = [decode_token(t) for t in tokens2]

print("\nWITH GPT-2 pre-tokenization:")
print(f"  Raw tokens ({len(tokens2)}): {tokens2}")
print(f"  Decoded: {decoded_tokens2}")
print(f"  Reconstructed: {''.join(decoded_tokens2)}")


print("\n" + "=" * 70)
print("Pre-tokenization Chunks Analysis")
print("=" * 70)

print("\nNONE mode chunks:")
chunks_none = normalizer_none.pre_tokenize(test_sentence)
print(f"  {chunks_none}")

print("\nGPT2 mode chunks:")
chunks_gpt2 = normalizer_gpt2.pre_tokenize(test_sentence)
print(f"  {chunks_gpt2}")


print("\n" + "=" * 70)
print("Top 10 Merges Learned (DECODED)")
print("=" * 70)

print("\nWITHOUT pre-tokenization:")
if len(merges1) > 0:
    for i, (left, right) in enumerate(merges1.merges[:10]):
        left_dec = decode_token(left)
        right_dec = decode_token(right)
        result_dec = decode_token(left + right)
        print(f"  {i+1}. '{left_dec}' + '{right_dec}' → '{result_dec}'")
else:
    print("  (No merges learned)")

print("\nWITH GPT-2 pre-tokenization:")
if len(merges2) > 0:
    for i, (left, right) in enumerate(merges2.merges[:10]):
        left_dec = decode_token(left)
        right_dec = decode_token(right)
        result_dec = decode_token(left + right)
        print(f"  {i+1}. '{left_dec}' + '{right_dec}' → '{result_dec}'")
else:
    print("  (No merges learned)")



print("\n3. Chunk boundaries (GPT-2):")
for i, chunk in enumerate(chunks_gpt2):
    print(f"   Chunk {i+1}: '{chunk}'")
