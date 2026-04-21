Day 27 completed successfully.

Outputs
- artifacts/day27/mimiciv_train_diffusion.pt
- artifacts/day27/mimiciv_val_diffusion.pt
- artifacts/day27/mimiciv_train_diffusion_summary.json
- artifacts/day27/mimiciv_val_diffusion_summary.json

Key checks
- train and val tensors built successfully
- attention masks and lengths verified
- lazy multi-hot batch view verified
- no empty records dropped
- max_len=256 truncation is active for long sequences

Important environment note
- current terminal used KMP_DUPLICATE_LIB_OK=TRUE as a temporary workaround
- before serious diffusion training, duplicate OpenMP runtime conflict should be cleaned permanently
