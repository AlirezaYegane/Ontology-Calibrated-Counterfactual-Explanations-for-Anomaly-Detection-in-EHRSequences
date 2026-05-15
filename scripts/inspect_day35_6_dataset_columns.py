import pandas as pd

paths = [
    r"data\processed\mimiciv_train_detector_supervised.pkl",
    r"data\processed\mimiciv_val_detector_supervised.pkl",
]

for p in paths:
    df = pd.read_pickle(p)
    print("\n==============================")
    print(p)
    print("shape:", df.shape)
    print("columns:")
    for c in df.columns:
        print(" -", c)
    print("\nlabel counts:")
    print(df["label"].value_counts(dropna=False) if "label" in df.columns else "no label")
    print("\nanomaly_type counts:")
    print(df["anomaly_type"].value_counts(dropna=False).head(20) if "anomaly_type" in df.columns else "no anomaly_type")
    print("\nhead:")
    print(df.head(3).to_string())
