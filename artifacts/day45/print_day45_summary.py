import json
from pathlib import Path

roots = [
    Path("artifacts/day45/detector_only/day45_test_set_metrics.json"),
    Path("artifacts/day45/no_ontology/day45_test_set_metrics.json"),
    Path("artifacts/day45/generative_only/day45_test_set_metrics.json"),
]

for p in roots:
    s = json.load(open(p, encoding="utf-8"))
    t = s["selected_threshold"]
    print("\n===", p.parent.name, "===")
    print("n_records:", s["n_records"])
    print("ROC-AUC:", s["global_metrics"]["roc_auc"])
    print("Average Precision:", s["global_metrics"]["average_precision"])
    print("Threshold:", t["threshold"])
    print("Precision:", t["precision"])
    print("Recall:", t["recall"])
    print("F1:", t["f1"])
    print("FPR:", t["false_positive_rate"])
    print("FNR:", t["false_negative_rate"])
