from bytepiece.core.normalizer import Normalizer, PreTokenizationMode
from bytepiece.core.code_pretokenizer import code_pretokenize


print("Python Code-Aware Pre-tokenization")
print("=" * 70)

# Test 1: String with URL
print("\nTest 1: URL in string literal")
code = 'url = "https://api.example.com/v1/users"'
result = code_pretokenize(code)
print(f"Input:  {code}")
print(f"Output: {result}")
assert '"https://api.example.com/v1/users"' in result, "URL should stay intact!"
print("✓ PASSED - URL preserved as single token")

# Test 2: Operators
print("\n Test 2: Comparison operators")
code = 'if x >= 100 and y <= 50:'
result = code_pretokenize(code)
print(f"Input:  {code}")
print(f"Output: {result}")
assert '>=' in result, ">= should be one token!"
assert '<=' in result, "<= should be one token!"
print("✓ PASSED - Operators preserved")

# Test 3: Comments
print("\n Test 3: Comments")
code = 'x = 5  # This is a comment'
result = code_pretokenize(code)
print(f"Input:  {code}")
print(f"Output: {result}")
assert '# This is a comment' in result, "Comment should be one chunk!"
print("✓ PASSED - Comment preserved")

# Test 4: Normalizer integration
print("\nTest 4: Normalizer CODE mode")
normalizer = Normalizer(pre_tokenization=PreTokenizationMode.CODE)
code = 'def hello(): return "world"'
result = normalizer.pre_tokenize(code)
print(f"Input:  {code}")
print(f"Output: {result}")
assert '"world"' in result, "String should be preserved!"
print("✓ PASSED - Normalizer integration works")

# Test 5: Empty and edge cases
print("\nTest 5: Edge cases")
assert code_pretokenize('') == [], "Empty string should return empty list"
assert len(code_pretokenize('   ')) > 0, "Whitespace should be preserved"
print("✓ PASSED - Edge cases handled")


print()