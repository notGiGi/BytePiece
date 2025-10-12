# BytePiece 🔡

> Educational, production-grade BPE tokenizer in pure Python
---

##  Quick Start

```bash
# Install
pip install bytepiece

# Train a tokenizer
bytepiece train bpe my_corpus.txt --vocab-size 5000

# Tokenize text
bytepiece apply model.json input.txt --output tokens.txt

# Inspect vocabulary
bytepiece inspect model.json --top-merges 20
```

---

##  Features

BytePiece implements **Byte Pair Encoding (BPE)** from scratch with modern best practices:

- ✅ **Complete BPE Implementation** - Training and inference in pure Python
- ✅ **Unicode Normalization** - NFKC/NFC/NFD support with configurable modes
- ✅ **Byte-Fallback** - 100% coverage for any Unicode text
- ✅ **Special Tokens** - `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` support
- ✅ **Syntax-Aware Tokenization** - Pre-tokenization for Python code
- ✅ **CLI + Python API** - Use from command line or programmatically
- ✅ **Reproducible Models** - Deterministic training with model hashing
- ✅ **Tested & Typed** - 65% test coverage, mypy strict mode

---

##  Why BytePiece?

- **Readable Code** - Understand how BPE actually works under the hood
- **Production Patterns** - Proper error handling, I/O, testing, documentation
- **Research-Ready** - Reproducible, extensible, benchmarked

---


## 📖 Usage

### Python API

```python
import bytepiece

# Train a tokenizer
encoder = bytepiece.train_bpe(
    corpus_path="data/corpus.txt",
    vocab_size=5000,
    seed=42,  # Reproducible
)

# Save model
bytepiece.save_model(encoder, "tokenizer.json")

# Load and use
encoder = bytepiece.load_model("tokenizer.json")

# Tokenize
text = "Hello, world! 你好 🌍"
tokens = encoder.encode(text)
# ['▁Hello', ',', '▁world', '!', '▁', '<0xE4>', '<0xBD>', '<0xA0>', ...]

# Decode
decoded = encoder.decode(tokens)
# 'Hello, world! 你好 🌍'
```

### Code Tokenization

```python
from bytepiece import train_bpe, Normalizer


encoder = train_bpe(
    corpus_path="python_code.txt",
    vocab_size=10000,
    normalizer=Normalizer(
        pre_tokenization_mode="python"  
    )
)

code = '''
def fibonacci(n):
    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
'''

tokens = encoder.encode(code)
```

### CLI

```bash
# Train
bytepiece train bpe corpus.txt \
    --vocab-size 5000 \
    --normalizer NFKC \
    --output model.json

# Apply
bytepiece apply model.json input.txt --output tokens.txt

# Inspect
bytepiece inspect model.json --top-merges 50

# Explain step-by-step
bytepiece explain "Hello, world!" model.json
```

---



### Reproducible Training

```python
# Same seed = same vocabulary
encoder1 = train_bpe("corpus.txt", vocab_size=5000, seed=42)
encoder2 = train_bpe("corpus.txt", vocab_size=5000, seed=42)

assert encoder1.vocab.tokens == encoder2.vocab.tokens
```

---

## 📚 Examples

See [`examples/`](examples/) directory:

- **[quickstart.py](examples/quickstart.py)** - Train and use in 5 minutes
- **[code_tokenization.py](examples/code_tokenization.py)** - Tokenizing Python code
- **[compare_normalizations.py](examples/compare_normalizations.py)** - Unicode normalization modes
- **[benchmarking.py](examples/benchmarking.py)** - Compare with SentencePiece


##  Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

Areas for contribution:
- Additional pre-tokenization strategies
- Unigram algorithm implementation
- Export/import for SentencePiece/HuggingFace formats
- Performance optimizations
- More comprehensive benchmarks

---

## 📄 License

Apache-2.0 - see [LICENSE](LICENSE)

