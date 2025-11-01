"""
Test de generalización: ¿El tokenizer memoriza o generaliza?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.core.io import load_model

# Cargar modelo Parity-Aware entrenado
# (Asume que el benchmark guardó el modelo)

print("="*60)
print("TEST DE GENERALIZACIÓN")
print("="*60)

# Textos NO vistos en training (diferentes combinaciones)
test_cases = {
    'Finnish': [
        "kissa koira talo",  # 3 palabras básicas
        "iso pieni vesi",
        "kirja juosta",
    ],
    'Turkish': [
        "kedi köpek ev",
        "büyük küçük su",
        "kitap koş",
    ],
    'English': [
        "cat dog house",
        "big small water",
        "book run",
    ],
    'German': [
        "Katze Hund Haus",
        "groß klein Wasser",
        "Buch laufen",
    ]
}

# Cargar modelo desde benchmark
import json
with open('benchmarks/results/parity_aware_comparison.json') as f:
    results = json.load(f)

# Reconstruir encoder (necesitamos hacerlo manual porque no guardamos el modelo)
print("\n⚠️  Necesitamos cargar el encoder entrenado.")
print("Por ahora, solo reportamos las métricas del benchmark:\n")

baseline_fert = results['baseline']['stats']['per_language']
parity_fert = results['parity_aware']['stats']['per_language']

print("\nFERTILITY COMPARISON:")
print(f"{'Language':<12} | {'Baseline':<10} | {'Parity':<10} | {'Change':<10}")
print("-"*50)

for lang in baseline_fert.keys():
    b = baseline_fert[lang]['fertility']
    p = parity_fert[lang]['fertility']
    change = (p - b) / b * 100
    print(f"{lang:<12} | {b:<10.4f} | {p:<10.4f} | {change:+.1f}%")

print("\n" + "="*60)
print("DIAGNÓSTICO:")
print("="*60)

# Check si todas las fertilidades son iguales (señal de overfitting)
parity_ferts = [parity_fert[lang]['fertility'] for lang in parity_fert]
std = (max(parity_ferts) - min(parity_ferts))

if std < 0.01:
    print("\n🚨 PROBLEMA DETECTADO: Overfitting")
    print(f"   Todas las fertilidades son idénticas (std={std:.4f})")
    print("   El modelo memoriza frases completas del training set.")
    print("\n   Solución:")
    print("   1. Limitar longitud de merges (max 15-20 caracteres)")
    print("   2. Reducir vocab_size (2000 → 1000)")
    print("   3. Aumentar diversidad del corpus")
else:
    print(f"\n✓ OK: Fertilidades tienen variación natural (std={std:.4f})")

# Check mejora realista
gini_improvement = (1 - results['parity_aware']['stats']['gini_coefficient'] / 
                    results['baseline']['stats']['gini_coefficient']) * 100

if gini_improvement > 90:
    print(f"\n🚨 SOSPECHOSO: Mejora de Gini demasiado perfecta ({gini_improvement:.1f}%)")
    print("   Valores realistas según papers: 50-70%")
    print("   Tu resultado: probablemente overfitting")
else:
    print(f"\n✓ OK: Mejora de Gini realista ({gini_improvement:.1f}%)")

print("\n" + "="*60)