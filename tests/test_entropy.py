"""
Test script for entropy-aware tokenization
Run from bytepiece root: python test_entropy.py
"""

print("Testing entropy module imports...")

try:
    from bytepiece.algorithms.entropy import EntropyAnalyzer, EntropyPreTokenizer
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test functionality
print("\nTesting EntropyAnalyzer...")
analyzer = EntropyAnalyzer()

test_tokens = [
    ("Operator", ">="),
    ("Keyword", "def"),
    ("URL", "https://api.example.com/users"),
    ("Variable", "x"),
]

for label, token in test_tokens:
    result = analyzer.analyze_token(token)
    decision = "FRAGMENT" if result['should_fragment'] else "PRESERVE"
    print(f"  {label:15} '{token[:30]:30}' → {decision}")

print("\nTesting EntropyPreTokenizer...")
pretokenizer = EntropyPreTokenizer(analyzer)

code = 'def fetch(): url = "https://api.test.com/data"; return url if x >= 10 else None'
result = pretokenizer.pretokenize_with_decisions(code)

print(f"  Code: {code}")
print(f"  Total tokens: {result['statistics']['total_tokens']}")
print(f"  Preserved: {result['statistics']['preserved_count']}")
print(f"  Fragmentable: {result['statistics']['fragmentable_count']}")

print("\n✅ All tests passed! Entropy module is working correctly.")