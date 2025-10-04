"""BytePiece quickstart example."""

import bytepiece

def main():
    print("=" * 60)
    print("BytePiece Quickstart Example")
    print("=" * 60)
    
    # Sample training data
    texts = [
        "Hello world!",
        "Hello BytePiece!",
        "This is a tokenizer example.",
        "We can handle Unicode: 你好世界",
        "And emojis too! 🚀",
    ]
    
    print("\n📚 Training corpus:")
    for text in texts:
        print(f"  - {text}")
    
    # Train a BPE tokenizer
    print("\n🔧 Training BPE tokenizer...")
    vocab, merges, normalizer = bytepiece.train_bpe(
        texts=texts,
        vocab_size=500,
        byte_fallback=True,
        verbose=False,
    )
    
    print(f"✓ Vocabulary size: {len(vocab)}")
    print(f"✓ Number of merges: {len(merges)}")
    
    # Create encoder
    encoder = bytepiece.BPEEncoder(vocab, merges, normalizer)
    
    # Test encoding
    print("\n🔤 Tokenization examples:")
    test_texts = [
        "Hello world!",
        "你好世界",
        "Unseen text gets tokenized too!",
    ]
    
    for text in test_texts:
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        print(f"\n  Input:   {text}")
        print(f"  Tokens:  {' '.join(tokens)}")
        print(f"  Decoded: {decoded}")
        print(f"  # tokens: {len(tokens)}")
    
    # Save model
    print("\n💾 Saving model...")
    bytepiece.save_model(encoder, "example_model.json")
    print("✓ Model saved to example_model.json")
    
    # Load model
    print("\n📂 Loading model...")
    loaded_encoder = bytepiece.load_model("example_model.json")
    print("✓ Model loaded successfully")
    
    # Verify it works
    text = "Testing loaded model"
    original_tokens = encoder.encode(text)
    loaded_tokens = loaded_encoder.encode(text)
    
    assert original_tokens == loaded_tokens, "Mismatch!"
    print(f"✓ Loaded model works correctly")
    
    # Show model info
    print("\n📊 Model information:")
    info = bytepiece.get_model_info("example_model.json")
    print(f"  Algorithm: {info['algorithm']}")
    print(f"  Version: {info['version']}")
    print(f"  Vocab size: {info['vocab_size']}")
    print(f"  Model hash: {info['model_hash'][:16]}...")
    
    print("\n" + "=" * 60)
    print("✅ Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
