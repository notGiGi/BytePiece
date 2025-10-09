#!/usr/bin/env python3
"""Prepare datasets for benchmarking.

Downloads real Python code from popular GitHub repositories
and prepares datasets for tokenization benchmarks.

Usage:
    python benchmarks/scripts/1_prepare_datasets.py
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple
import random

# Configuration
BENCHMARK_ROOT = Path(__file__).parent.parent
DATASET_DIR = BENCHMARK_ROOT / "datasets"
PYTHON_DIR = DATASET_DIR / "python"
ENGLISH_DIR = DATASET_DIR / "english"

# Popular Python repos (small, well-written code)
PYTHON_REPOS = [
    ("https://github.com/psf/requests", "requests"),
    ("https://github.com/pallets/flask", "flask"),
    ("https://github.com/pytest-dev/pytest", "pytest"),
    ("https://github.com/python/cpython", "cpython"),
    ("https://github.com/django/django", "django"),
]

# File size filters (lines)
MIN_LINES = 10
MAX_LINES = 500
TARGET_FILES = 200  # Target number of files per dataset


def setup_directories():
    """Create benchmark directory structure."""
    print("📁 Setting up directories...")
    
    dirs = [
        DATASET_DIR,
        PYTHON_DIR,
        ENGLISH_DIR,
        BENCHMARK_ROOT / "models",
        BENCHMARK_ROOT / "results" / "raw",
        BENCHMARK_ROOT / "results" / "plots",
        BENCHMARK_ROOT / "results" / "reports",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print()


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0


def clone_repo(url: str, name: str) -> Path:
    """Clone a GitHub repository."""
    temp_dir = DATASET_DIR / "temp" / name
    
    if temp_dir.exists():
        print(f"  Repository {name} already cloned, skipping...")
        return temp_dir
    
    print(f"  Cloning {name}...")
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(temp_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"    ✓ Cloned {name}")
    except subprocess.CalledProcessError:
        print(f"    ✗ Failed to clone {name}")
        return None
    
    return temp_dir


def extract_python_files(repo_dir: Path, repo_name: str) -> List[Path]:
    """Extract Python files from repository."""
    if repo_dir is None:
        return []
    
    py_files = []
    
    # Find all .py files
    for py_file in repo_dir.rglob("*.py"):
        # Skip tests, docs, examples
        if any(x in str(py_file) for x in ["test", "example", "doc", "__pycache__"]):
            continue
        
        # Check file size
        num_lines = count_lines(py_file)
        if MIN_LINES <= num_lines <= MAX_LINES:
            py_files.append((py_file, num_lines))
    
    # Sort by number of lines for consistency
    py_files.sort(key=lambda x: x[1])
    
    # Sample evenly across file sizes
    if len(py_files) > TARGET_FILES // len(PYTHON_REPOS):
        step = len(py_files) // (TARGET_FILES // len(PYTHON_REPOS))
        py_files = py_files[::step]
    
    print(f"    Found {len(py_files)} suitable Python files")
    
    # Copy to dataset directory
    copied = []
    for i, (py_file, num_lines) in enumerate(py_files):
        target = PYTHON_DIR / f"{repo_name}_{i:03d}_{num_lines}lines.py"
        shutil.copy2(py_file, target)
        copied.append(target)
    
    return copied


def prepare_python_dataset():
    """Prepare Python code dataset from GitHub repos."""
    print("🐍 Preparing Python dataset...")
    print(f"   Target: {TARGET_FILES} files from {len(PYTHON_REPOS)} repositories")
    print()
    
    all_files = []
    
    for url, name in PYTHON_REPOS:
        print(f"  Processing {name}...")
        repo_dir = clone_repo(url, name)
        files = extract_python_files(repo_dir, name)
        all_files.extend(files)
    
    print(f"\n  ✓ Collected {len(all_files)} Python files")
    
    # Calculate statistics
    total_chars = 0
    total_lines = 0
    
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            total_chars += len(content)
            total_lines += content.count('\n')
    
    print(f"  Total: {total_chars:,} characters, {total_lines:,} lines")
    print(f"  Average: {total_chars // len(all_files):,} chars/file")
    print()
    
    # Cleanup temp directory
    # Cleanup temp directory (Windows-safe)
    temp_dir = DATASET_DIR / "temp"
    if temp_dir.exists():
        print("  Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print("  ⚠️  Could not delete temp files (Windows permission issue)")
            print("  You can manually delete: benchmarks\\datasets\\temp\\")
        
        return all_files


def create_english_dataset():
    """Create a simple English dataset from sample text."""
    print("📝 Creating English text dataset...")
    
    # Sample English sentences (in real scenario, would download from Wikipedia)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "Tokenization is the process of breaking text into tokens.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "The internet has revolutionized how we communicate and share information.",
        "Climate change is one of the most pressing issues of our time.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Quantum computing promises to solve problems beyond classical computers.",
    ] * 20  # Repeat to create more data
    
    # Write to files
    for i, text in enumerate(sample_texts):
        target = ENGLISH_DIR / f"english_{i:03d}.txt"
        with open(target, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"  ✓ Created {len(sample_texts)} English text files")
    print(f"  Note: This is sample data. For real benchmarks, use Wikipedia/OSCAR.")
    print()


def create_dataset_manifest():
    """Create a manifest of all datasets."""
    print("📋 Creating dataset manifest...")
    
    manifest_path = DATASET_DIR / "manifest.txt"
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("BytePiece Benchmark Datasets\n")
        f.write("=" * 50 + "\n\n")
        
        # Python files
        python_files = list(PYTHON_DIR.glob("*.py"))
        f.write(f"Python Code: {len(python_files)} files\n")
        f.write(f"  Location: {PYTHON_DIR}\n")
        
        total_size = sum(p.stat().st_size for p in python_files)
        f.write(f"  Total size: {total_size / 1024:.1f} KB\n\n")
        
        # English files
        english_files = list(ENGLISH_DIR.glob("*.txt"))
        f.write(f"English Text: {len(english_files)} files\n")
        f.write(f"  Location: {ENGLISH_DIR}\n")
        
        total_size = sum(p.stat().st_size for p in english_files)
        f.write(f"  Total size: {total_size / 1024:.1f} KB\n\n")
    
    print(f"  ✓ Manifest saved to {manifest_path}")
    print()


def main():
    """Main entrypoint."""
    print("\n" + "=" * 70)
    print("📊 BytePiece Benchmark Dataset Preparation")
    print("=" * 70 + "\n")
    
    # Setup
    setup_directories()
    
    # Prepare datasets
    python_files = prepare_python_dataset()
    create_english_dataset()
    
    # Create manifest
    create_dataset_manifest()
    
    # Summary
    print("=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. python benchmarks/scripts/2_train_models.py")
    print("  2. python benchmarks/scripts/3_run_benchmarks.py")
    print("  3. python benchmarks/scripts/4_generate_report.py")
    print()


if __name__ == "__main__":
    main()