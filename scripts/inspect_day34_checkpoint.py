from __future__ import annotations

import inspect
from pathlib import Path

import torch

from src.models.diffusion import DiffusionModel


ckpt_path = Path(r"D:\Article\outputs\diffusion\day33_ontology_regularized_fixed\last.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("CHECKPOINT:", ckpt_path)
print("CKPT TYPE:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nTOP-LEVEL KEYS:")
    for k in ckpt.keys():
        print(" -", k)

    print("\nCONFIG:")
    cfg = ckpt.get("config", None)
    print(cfg)

    state = (
        ckpt.get("model_state")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
    )

    if state is None:
        print("\nNo standard model state key found.")
    else:
        print("\nSTATE TYPE:", type(state))
        print("NUM STATE KEYS:", len(state))

        print("\nFIRST 80 STATE KEYS:")
        for i, (k, v) in enumerate(state.items()):
            if i >= 80:
                break
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            print(f"{i:03d} {k} {shape}")

print("\nDiffusionModel signature:")
print(inspect.signature(DiffusionModel))
