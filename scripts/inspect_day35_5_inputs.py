import pandas as pd

paths = [
    r"data\processed\mimiciv_train_detector_supervised.pkl",
    r"data\processed\mimiciv_val_detector_supervised.pkl",
]

for p in paths:
    df = pd.read_pickle(p)
    print("\n==", p)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(2).to_string())
