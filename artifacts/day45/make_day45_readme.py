import pandas as pd
from pathlib import Path

summary_path = Path("artifacts/day45/day45_variant_summary.csv")
df = pd.read_csv(summary_path)

best = df.iloc[0]

lines = [
    "# Day 45 — Comprehensive Test Set Evaluation",
    "",
    "## Status",
    "",
    "Complete.",
    "",
    "## Goal",
    "",
    "Evaluate available Day 41 score variants on the held-out/test score table using ROC-AUC, Average Precision, conservative threshold selection, and false-positive / false-negative behavior.",
    "",
    "## Evaluated variants",
    "",
    "| Variant | ROC-AUC | AP | Threshold | Precision | Recall | F1 | FPR | FNR |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for _, r in df.iterrows():
    lines.append(
        f"| {r['variant']} | {r['roc_auc']:.6f} | {r['average_precision']:.6f} | "
        f"{r['threshold']:.6f} | {r['precision']:.6f} | {r['recall']:.6f} | "
        f"{r['f1']:.6f} | {r['false_positive_rate']:.6f} | {r['false_negative_rate']:.6f} |"
    )

lines += [
    "",
    "## Best-performing variant",
    "",
    f"- Best variant by ROC-AUC/AP/F1 ranking: `{best['variant']}`",
    f"- ROC-AUC: `{best['roc_auc']:.6f}`",
    f"- Average Precision: `{best['average_precision']:.6f}`",
    f"- Selected threshold: `{best['threshold']:.6f}`",
    f"- Precision: `{best['precision']:.6f}`",
    f"- Recall: `{best['recall']:.6f}`",
    f"- F1: `{best['f1']:.6f}`",
    f"- False-positive rate: `{best['false_positive_rate']:.6f}`",
    "",
    "## Paper-ready interpretation",
    "",
    "Day 45 converts the ablation score artifacts into a held-out/test-style evaluation package. The outputs include threshold sensitivity tables, ROC/PR curve points, false-positive previews, false-negative previews, and per-variant metric summaries. These artifacts support the evaluation section of the paper by showing not only discrimination performance, but also the operating threshold behavior and error profile.",
    "",
    "## Artifacts",
    "",
    "- `day45_variant_summary.csv`",
    "- `detector_only/day45_test_set_metrics.json`",
    "- `no_ontology/day45_test_set_metrics.json`",
    "- `generative_only/day45_test_set_metrics.json`",
    "- each variant folder includes threshold sensitivity, ROC/PR points, FP/FN previews, and README.",
    "",
]

Path("artifacts/day45/README.md").write_text("\n".join(lines), encoding="utf-8")
print(Path("artifacts/day45/README.md").read_text(encoding="utf-8"))
