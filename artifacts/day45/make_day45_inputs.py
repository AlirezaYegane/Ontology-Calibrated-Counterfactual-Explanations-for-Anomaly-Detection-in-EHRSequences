import pandas as pd
from pathlib import Path

src = Path("artifacts/day41/day41_variant_scores.csv")
out = Path("artifacts/day45")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
variants = ["detector_only", "generative_only", "no_ontology"]

for variant in variants:
    if variant not in df.columns:
        print(f"SKIP missing variant: {variant}")
        continue

    clean = pd.DataFrame({
        "record_id": range(len(df)),
        "y_true": df["y_true"].astype(int),
        "score": pd.to_numeric(df[variant], errors="coerce"),
        "source_variant": variant,
    })

    out_path = out / f"day45_input_{variant}.csv"
    clean.to_csv(out_path, index=False)
    print(f"WROTE {out_path} shape={clean.shape} score_min={clean['score'].min():.6f} score_max={clean['score'].max():.6f}")
