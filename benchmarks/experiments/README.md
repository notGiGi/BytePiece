# Language Balance Experiments

Research experiments for measuring and addressing tokenization imbalance across languages.

## Experiments

### 01_language_imbalance.py
Initial validation experiment that measures fertility (tokens/word) across 6 languages
using the vanilla BPE implementation.

**Run:**
```bash
python benchmarks\experiments\01_language_imbalance.py
```

**Results:** Saved to `benchmarks\results\language_balance\01_initial_validation.json`

**Expected outcome:** Confirms whether language imbalance is significant (2x+ gap)


