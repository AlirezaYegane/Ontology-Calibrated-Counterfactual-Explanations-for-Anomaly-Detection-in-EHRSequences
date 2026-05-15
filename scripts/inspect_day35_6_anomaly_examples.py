import pandas as pd

df = pd.read_pickle(r"data\processed\mimiciv_val_detector_supervised.pkl")

seq_col = "sequence_tokens" if "sequence_tokens" in df.columns else "codes"

for anomaly_type in df["anomaly_type"].fillna("").unique():
    print("\n==============================")
    print("ANOMALY TYPE:", repr(anomaly_type))
    sub = df[df["anomaly_type"].fillna("") == anomaly_type].head(3)
    for i, row in sub.iterrows():
        print("\nrow", i, "label=", row.get("label"))
        toks = row[seq_col]
        print(toks[:80] if isinstance(toks, list) else str(toks)[:1000])
