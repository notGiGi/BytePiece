from bytepiece.core.normalizer import Normalizer, PreTokenizationMode

print("Test 1: NONE (no pre-tokenization)")

normalizer = Normalizer(pre_tokenization=PreTokenizationMode.NONE)
text = "Hello world!"
chunks = normalizer.pre_tokenize(text)
print(f"Text: '{text}'")
print(f"Chunks: {chunks}")
print(f"Number of chunks: {len(chunks)}")
print()


print("Test 2: WHITESPACE")

normalizer = Normalizer(pre_tokenization=PreTokenizationMode.WHITESPACE)
text = "Hello world!"
chunks = normalizer.pre_tokenize(text)
print(f"Text: '{text}'")
print(f"Chunks: {chunks}")
print(f"Number of chunks: {len(chunks)}")
print()


print("Test 3: GPT2")

normalizer = Normalizer(pre_tokenization=PreTokenizationMode.GPT2)

tests = [
    "Hello world!",
    "I don't know",
    "John's book",
    "Price: $99.99!",
    "Hello 你好 world",
    "They're going",
]

for text in tests:
    chunks = normalizer.pre_tokenize(text)
    print(f"Text: '{text}'")
    print(f"  → {chunks}")
    print()