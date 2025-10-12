"""Shared test fixtures and configuration for BytePiece tests."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_path():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Sample text for quick tests."""
    return "hello world python programming machine learning"


@pytest.fixture
def unicode_text():
    """Sample text with Unicode characters."""
    return "Hello 世界 café résumé 🌍 مرحبا"


@pytest.fixture
def code_sample():
    """Sample Python code."""
    return '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Result: {result}")
'''


@pytest.fixture
def small_corpus(tmp_path):
    """Create a small corpus file for quick tests."""
    corpus_path = tmp_path / "small_corpus.txt"
    corpus_path.write_text(
        "the quick brown fox\n"
        "jumps over the lazy dog\n"
        "the dog was not amused\n"
        "the fox ran away quickly\n"
    )
    return str(corpus_path)


@pytest.fixture
def medium_corpus(tmp_path):
    """Create a medium-sized corpus for more realistic tests."""
    corpus_path = tmp_path / "medium_corpus.txt"
    
    text = ""
    for i in range(100):
        text += f"This is sentence number {i} in the corpus.\n"
        text += "Python is a great programming language.\n"
        text += "Machine learning and AI are transforming technology.\n"
    
    corpus_path.write_text(text)
    return str(corpus_path)


@pytest.fixture
def multilingual_corpus(tmp_path):
    """Create a multilingual corpus."""
    corpus_path = tmp_path / "multilingual.txt"
    corpus_path.write_text(
        "Hello world\n"
        "Bonjour le monde\n"
        "Hola mundo\n"
        "你好世界\n"
        "مرحبا بالعالم\n"
    )
    return str(corpus_path)


# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )