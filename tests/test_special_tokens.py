"""Test special tokens functionality."""
from bytepiece import train_bpe, BPEEncoder, SpecialTokens

# Test corpus
texts = [
    "Hello world!",
    "Special tokens test",
]

print("=" * 70)
print("Testing Special Tokens")
print("=" * 70)

# Train WITH special tokens
print("\n1. Training with special tokens...")
vocab, merges, normalizer = train_bpe(
    texts=texts,
    vocab_size=500,
    use_special_tokens=True,  # ← Enable special tokens
    verbose=True,
)

encoder = BPEEncoder(vocab, merges, normalizer)

# Check special tokens exist
print("\n2. Checking special tokens...")
print(f"  PAD token: {encoder.vocab.pad_token} (ID: {encoder.vocab.pad_token_id})")
print(f"  UNK token: {encoder.vocab.unk_token} (ID: {encoder.vocab.unk_token_id})")
print(f"  BOS token: {encoder.vocab.bos_token} (ID: {encoder.vocab.bos_token_id})")
print(f"  EOS token: {encoder.vocab.eos_token} (ID: {encoder.vocab.eos_token_id})")

# Verify they're at the beginning of vocab
assert encoder.vocab.pad_token_id == 0, "PAD should be ID 0"
assert encoder.vocab.unk_token_id == 1, "UNK should be ID 1"
assert encoder.vocab.bos_token_id == 2, "BOS should be ID 2"
assert encoder.vocab.eos_token_id == 3, "EOS should be ID 3"
print("  ✓ Special tokens have correct IDs")

# Test encoding with BOS/EOS
print("\n3. Testing encoding with BOS/EOS...")
text = "Hello world"

# Manual add BOS/EOS
tokens = encoder.encode(text)
tokens_with_special = [SpecialTokens.BOS] + tokens + [SpecialTokens.EOS]
print(f"  Original tokens: {tokens[:5]}...")
print(f"  With BOS/EOS: {tokens_with_special[:3]} ... {tokens_with_special[-3:]}")

# Test decoding (special tokens should be removed)
decoded = encoder.decode(tokens_with_special)
print(f"  Decoded: '{decoded}'")

# Verify special tokens were actually removed
assert "<BOS>" not in decoded, "BOS should be removed during decoding"
assert "<EOS>" not in decoded, "EOS should be removed during decoding"
assert decoded.strip() == "Hello world", "Decoded text should match original (without normalization)"
print(f"  ✓ Decoding correctly removes special tokens")

# Test vocab size
print("\n4. Vocab size comparison...")
vocab_without, _, _ = train_bpe(texts, 500, use_special_tokens=False)
print(f"  Without special tokens: {len(vocab_without)}")
print(f"  With special tokens: {len(vocab)}")
print(f"  Difference: {len(vocab) - len(vocab_without)} (should be 4)")

assert len(vocab) == len(vocab_without) + 4, "Should have 4 more tokens"


