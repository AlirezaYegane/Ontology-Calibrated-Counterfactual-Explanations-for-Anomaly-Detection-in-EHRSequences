# Day 23: Anomaly Detector Design

## Model Architecture

The anomaly detector is a GRU-based autoregressive model defined in
`src/models/detector.py` (`AnomalyDetectorGRU`).

```
Input (B, L) integer token indices
  |
  v
Embedding(vocab_size, 128, padding_idx=0)  ->  (B, L, 128)
  |
  v
GRU(128, 128, num_layers=1, batch_first=True)  ->  (B, L, 128)
  |
  v
Dropout(0.2)
  |
  v
Linear(128, vocab_size)  ->  (B, L, V) logits
```

The embedding layer maps each clinical code token to a 128-dimensional
vector.  A single-layer GRU processes the sequence left-to-right.
Dropout is applied to the GRU output, and a linear projection produces
per-token logits over the full vocabulary.

## Training Objective

The model is trained with **next-token prediction** using cross-entropy
loss.  Given input sequence `x = [x_0, x_1, ..., x_L]`, the model
predicts `x[t+1]` from the hidden state at position `t`:

- Input to the model: `x[:, :-1]` (all tokens except the last)
- Target: `x[:, 1:]` (all tokens except the first)
- Loss: `F.cross_entropy(logits, targets, ignore_index=PAD)`

Sequences are prepended with a `<BOS>` token and appended with `<EOS>`.
Padding tokens are masked from the loss via `ignore_index`.

## Anomaly Scoring

At inference time the anomaly score for a sequence is the **mean
negative log-likelihood (NLL)** across all non-padding positions:

```
score(x) = (1 / |non-pad positions|) * sum_t(-log P(x[t+1] | x[0:t]))
```

Higher scores indicate sequences that the model finds harder to predict,
suggesting the sequence deviates from the patterns learned during training.

## Hyperparameters

All hyperparameters are stored in `config/detector.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embed_dim` | 128 | Sufficient for clinical code vocabularies of ~10k tokens |
| `hidden_dim` | 128 | Matched to embedding dim; keeps model lightweight |
| `num_layers` | 1 | Single layer avoids overfitting on moderate-size datasets |
| `dropout` | 0.2 | Standard regularization for sequence models |
| `batch_size` | 64 | Fits in single-GPU memory (8-16 GB VRAM) |
| `learning_rate` | 0.001 | Adam default; effective for GRU training |
| `weight_decay` | 0.0001 | Light L2 regularization |
| `max_seq_len` | 256 | Covers >95th percentile of admission sequence lengths |
| `patience` | 5 | Early stopping on validation loss to prevent overfitting |
| `epochs` | 30 | Upper bound; early stopping typically triggers sooner |
| `seed` | 42 | Reproducibility |

The model is intentionally lightweight (under 500k parameters for a 5k
vocabulary) to enable rapid iteration on a single consumer GPU.

## Synthetic Anomaly Types

Evaluation uses a labeled test set built by `src/preprocessing/anomaly_injection.py`
with three types of injected anomalies:

| Type | Method | Clinical analogue |
|------|--------|-------------------|
| `missing_indication` | Remove all `SNOMED:` diagnosis tokens | Medication prescribed without any recorded diagnosis |
| `random_code_swap` | Replace one `SNOMED:` token with a random code | Miscoded diagnosis or data entry error |
| `demographic_conflict` | Add a pregnancy code to a male patient record | Demographic-inconsistent coding |

Each type generates `n_per_type` anomalous records (default 500).  An
equal number of unmodified records serve as the normal class.

## Evaluation Metrics

The evaluation script (`src/evaluation/evaluate_detector.py`) computes:

| Metric | Description |
|--------|-------------|
| **ROC AUC** | Area under the receiver operating characteristic curve; measures ranking quality across all thresholds |
| **Average Precision (AP)** | Area under the precision-recall curve; more informative than ROC AUC when the anomalous class is small |
| **Precision @ p80** | Precision when the threshold is set at the 80th percentile of all scores |
| **Recall @ p80** | Recall at the same threshold |
| **F1 @ p80** | Harmonic mean of precision and recall at that threshold |
| **Mean score (normal)** | Average anomaly score for unmodified records |
| **Mean score (anomalous)** | Average anomaly score for injected anomalies |

Results are saved to `logs/detector/eval_results.json`.

## Frequency Baseline

The frequency baseline (`src/evaluation/frequency_baseline.py`) provides
a non-learned comparison.  It scores each sequence as:

```
score(x) = mean_t(-log(freq(x_t)))
```

where `freq(x_t)` is the relative frequency of token `x_t` in the
training set.  Tokens not seen during training receive a minimum
frequency of `1e-7`.

This baseline captures whether rare tokens drive the anomaly signal.
The GRU detector should outperform it by also capturing sequential
dependencies and co-occurrence patterns.

Results are saved to `logs/detector/frequency_baseline.json`.
