"""
Script to update bytepiece/__init__.py to export Fair BPE.

Run this after adding fair_bpe.py to bytepiece/algorithms/
"""

import sys
from pathlib import Path


def update_init_file():
    """Update bytepiece/__init__.py to include fair_bpe exports."""
    
    # Find bytepiece/__init__.py
    init_path = Path("bytepiece") / "__init__.py"
    
    if not init_path.exists():
        print(f"❌ ERROR: {init_path} not found!")
        print("   Make sure you're running this from the project root.")
        return 1
    
    print(f"📝 Updating {init_path}...")
    
    # Read current content
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'fair_bpe' in content:
        print("✓ Fair BPE already exported in __init__.py")
        return 0
    
    # Find the import section
    import_line = "from bytepiece.algorithms.bpe import BPEEncoder, train_bpe"
    
    if import_line not in content:
        print(f"❌ ERROR: Expected import line not found:")
        print(f"   {import_line}")
        return 1
    
    # Add fair_bpe import
    new_import = "from bytepiece.algorithms.fair_bpe import FairBPE, train_fair_bpe"
    content = content.replace(
        import_line,
        f"{import_line}\n{new_import}"
    )
    
    # Update __all__
    all_line = "__all__ = ["
    if all_line in content:
        # Find the __all__ list and add our exports
        lines = content.split('\n')
        updated_lines = []
        in_all = False
        
        for line in lines:
            updated_lines.append(line)
            
            if '__all__ = [' in line:
                in_all = True
            
            if in_all and '"train_bpe"' in line:
                # Add fair_bpe exports after train_bpe
                indent = line[:len(line) - len(line.lstrip())]
                updated_lines.append(f'{indent}"train_fair_bpe",')
                updated_lines.append(f'{indent}"FairBPE",')
            
            if in_all and ']' in line:
                in_all = False
        
        content = '\n'.join(updated_lines)
    
    # Write updated content
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated __init__.py")
    print("\nAdded exports:")
    print("  - train_fair_bpe")
    print("  - FairBPE")
    
    return 0


def verify_installation():
    """Verify that Fair BPE can be imported."""
    print("\n🔍 Verifying installation...")
    
    try:
        from bytepiece.algorithms.fair_bpe import train_fair_bpe, FairBPE
        print("✅ Direct import: bytepiece.algorithms.fair_bpe ✓")
    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
        return 1
    
    try:
        from bytepiece import train_fair_bpe, FairBPE
        print("✅ Package import: bytepiece.train_fair_bpe ✓")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return 1
    
    print("\n✨ Fair BPE is ready to use!")
    print("\nExample:")
    print("  from bytepiece import train_fair_bpe")
    print("  vocab, merges, norm, stats = train_fair_bpe(...)")
    
    return 0


def main():
    print("=" * 80)
    print("Fair BPE Integration Script")
    print("=" * 80)
    print()
    
    # Check that fair_bpe.py exists
    fair_bpe_path = Path("bytepiece") / "algorithms" / "fair_bpe.py"
    
    if not fair_bpe_path.exists():
        print(f"❌ ERROR: {fair_bpe_path} not found!")
        print()
        print("Please copy fair_bpe.py to bytepiece/algorithms/ first:")
        print("  cp fair_bpe.py bytepiece/algorithms/")
        return 1
    
    print(f"✓ Found {fair_bpe_path}")
    print()
    
    # Update __init__.py
    result = update_init_file()
    if result != 0:
        return result
    
    # Verify
    result = verify_installation()
    
    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Run test: python test_fair_bpe_fixed.py")
    print("  2. Run benchmark: python benchmarks/benchmark_fairness.py")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    sys.exit(main())