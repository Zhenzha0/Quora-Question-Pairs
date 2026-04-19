import time
import os
import csv

from dotenv import load_dotenv
load_dotenv()  # loads KAGGLE_USERNAME and KAGGLE_KEY from .env

import kagglehub
import numpy as np
import torch
import zarr
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_FILE = "cross_encoder_scores.zarr"
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ---------------------------------------------------------------------------
# Load dataset — collect all pairs (index, qid1, qid2, q1_text, q2_text)
# ---------------------------------------------------------------------------
print("[INFO] Loading dataset...", flush=True)
path = kagglehub.dataset_download("quora/question-pairs-dataset")
print("Path to dataset files:", path)

# Find the CSV file
csv_file = os.path.join(path, "questions.csv")
if not os.path.exists(csv_file):
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            csv_file = os.path.join(path, fname)
            break

REQUIRED_HEADERS = {"id", "qid1", "qid2", "question1", "question2", "is_duplicate"}

# Check headers of the found file; fall back to any CSV with required headers
with open(csv_file, newline="", encoding="utf-8") as f:
    headers = set(csv.DictReader(f).fieldnames or [])
if not REQUIRED_HEADERS.issubset(headers):
    found = False
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            candidate = os.path.join(path, fname)
            with open(candidate, newline="", encoding="utf-8") as f:
                h = set(csv.DictReader(f).fieldnames or [])
            if REQUIRED_HEADERS.issubset(h):
                csv_file = candidate
                found = True
                break
    if not found:
        raise FileNotFoundError(
            f"No CSV with headers {sorted(REQUIRED_HEADERS)} found in {path}"
        )

indices: list[int] = []
qid1s:   list[int] = []
qid2s:   list[int] = []
pairs:   list[tuple[str, str]] = []  # (question1, question2)

with open(csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            idx  = int(row["id"])
            qid1 = int(row["qid1"])
            qid2 = int(row["qid2"])
            q1   = row["question1"] or ""
            q2   = row["question2"] or ""
        except (KeyError, ValueError, TypeError):
            continue
        indices.append(idx)
        qid1s.append(qid1)
        qid2s.append(qid2)
        pairs.append((q1, q2))

N = len(pairs)
print(f"[INFO] Total question pairs to encode: {N}", flush=True)

# ---------------------------------------------------------------------------
# Load cross-encoder model
# ---------------------------------------------------------------------------
print(f"[INFO] Loading cross-encoder model: {MODEL_NAME}", flush=True)
model = CrossEncoder(MODEL_NAME)

# Access the underlying HuggingFace model and tokenizer so we can run a
# single forward pass that yields BOTH the scalar relevance score AND the
# full CLS-token hidden-state embedding.
auto_model = model.model       # AutoModelForSequenceClassification
tokenizer  = model.tokenizer

device = next(auto_model.parameters()).device
print(f"[INFO] Model device: {device}", flush=True)

# Determine hidden size with a tiny dummy forward pass.
with torch.no_grad():
    _dummy_enc = tokenizer(
        [("probe", "probe")],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8,
    )
    _dummy_enc = {k: v.to(device) for k, v in _dummy_enc.items()}
    _dummy_out = auto_model(**_dummy_enc, output_hidden_states=True)
    hidden_size: int = _dummy_out.hidden_states[-1].shape[-1]
    del _dummy_enc, _dummy_out

print(f"[INFO] Cross-encoder hidden size: {hidden_size}", flush=True)

# ---------------------------------------------------------------------------
# Open zarr store
# ---------------------------------------------------------------------------
store = zarr.open(OUTPUT_FILE, mode="w")

# index array: row index from the original CSV
index_arr = store.zeros(
    name="index",
    shape=(N,),
    dtype="int64",
    chunks=(BATCH_SIZE,),
)
index_arr[:] = np.array(indices, dtype=np.int64)

# qid1 / qid2 arrays
qid1_arr = store.zeros(name="qid1", shape=(N,), dtype="int64", chunks=(BATCH_SIZE,))
qid1_arr[:] = np.array(qid1s, dtype=np.int64)

qid2_arr = store.zeros(name="qid2", shape=(N,), dtype="int64", chunks=(BATCH_SIZE,))
qid2_arr[:] = np.array(qid2s, dtype=np.int64)

# cross_encoder_score: scalar sigmoid probability — kept for backwards compat
score_arr = store.full(
    name="cross_encoder_score",
    shape=(N,),
    fill_value=float("nan"),
    dtype="float32",
    chunks=(BATCH_SIZE,),
)

# cross_encoder_features: CLS-token hidden state from the final layer.
# Shape (N, hidden_size).  This is the rich, multi-dimensional CE output
# that gru_model_v4 appends to its scalar bridge instead of the single score.
features_arr = store.zeros(
    name="cross_encoder_features",
    shape=(N, hidden_size),
    dtype="float32",
    chunks=(BATCH_SIZE, hidden_size),
)

# ---------------------------------------------------------------------------
# Score in batches
# ---------------------------------------------------------------------------
print(
    f"[INFO] Starting cross-encoder scoring of {N} pairs "
    f"in batches of {BATCH_SIZE}...",
    flush=True,
)
score_start = time.time()
last_log_time = score_start

for i in range(0, N, BATCH_SIZE):
    batch = pairs[i : i + BATCH_SIZE]
    texts = [(q1, q2) for q1, q2 in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = auto_model(**encoded, output_hidden_states=True)

    # CLS token from the last hidden layer → (batch, hidden_size)
    cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().float().numpy()

    # Scalar relevance score: sigmoid of the classification logit → (batch,)
    scores = torch.sigmoid(outputs.logits.squeeze(-1)).cpu().float().numpy()

    batch_len = len(batch)
    features_arr[i : i + batch_len] = cls_emb
    score_arr[i : i + batch_len]    = scores

    now = time.time()
    elapsed = now - score_start
    done = i + batch_len
    pct  = done / N * 100
    rate = done / elapsed if elapsed > 0 else 0
    eta  = (N - done) / rate if rate > 0 else 0

    if (now - last_log_time) >= 30 or i == 0:
        print(
            f"[PROGRESS] {done}/{N} pairs ({pct:.1f}%) | "
            f"Elapsed: {format_duration(elapsed)} | "
            f"ETA: {format_duration(eta)} | "
            f"Speed: {rate:.1f} pairs/s",
            flush=True,
        )
        last_log_time = now

total_time = time.time() - score_start
print(
    f"[DONE] Scoring complete in {format_duration(total_time)} "
    f"({N / total_time:.1f} pairs/s avg)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nSaved {N} cross-encoder outputs to {OUTPUT_FILE}")
print(f"  store['index']                 shape: {store['index'].shape}")
print(f"  store['qid1']                  shape: {store['qid1'].shape}")
print(f"  store['qid2']                  shape: {store['qid2'].shape}")
print(f"  store['cross_encoder_score']   shape: {store['cross_encoder_score'].shape}")
print(f"  store['cross_encoder_features'] shape: {store['cross_encoder_features'].shape}")
print()
print("Example lookup — first pair:")
print(f"  index                  : {store['index'][0]}")
print(f"  qid1                   : {store['qid1'][0]}")
print(f"  qid2                   : {store['qid2'][0]}")
print(f"  cross_encoder_score    : {store['cross_encoder_score'][0]:.6f}")
print(f"  cross_encoder_features : shape={store['cross_encoder_features'][0].shape}  "
      f"norm={np.linalg.norm(store['cross_encoder_features'][0]):.4f}")
