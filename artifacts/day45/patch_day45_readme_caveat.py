from pathlib import Path

p = Path("artifacts/day45/README.md")
text = p.read_text(encoding="utf-8")

insert = """
## Scientific caveat

The `detector_only` and `no_ontology` variants produce identical ROC-AUC, Average Precision, precision, recall, F1, FPR, and FNR after threshold calibration. This indicates that, in the available Day 41 wide-format score artifact, `no_ontology` preserves the same sample ranking as `detector_only`, likely through monotonic score scaling. Therefore, this evaluation supports detector-driven discrimination but should not be over-interpreted as proving that ontology information is ineffective.

The `generative_only` variant has no useful discrimination in this artifact. Its ROC-AUC of 0.500000 and FPR of 1.000000 arise because the available generative-only scores are constant at zero. This is useful negative evidence for the ablation section: the current generative-only score artifact does not provide independent anomaly ranking signal.

## Recommended paper wording

The detector-based score achieved ROC-AUC 0.8002 and Average Precision 0.7332 on the Day 45 held-out/test-style evaluation table. Under a conservative threshold selected with precision and false-positive constraints, it reached precision 0.8018, recall 0.5665, and FPR 0.0418. The no-ontology score showed identical ranking behaviour in this artifact, while the generative-only score was non-discriminative due to constant zero-valued scores.
"""

marker = "## Paper-ready interpretation"
if "## Scientific caveat" not in text:
    text = text.replace(marker, insert + "\n" + marker)

p.write_text(text, encoding="utf-8")
print(p.read_text(encoding="utf-8"))
