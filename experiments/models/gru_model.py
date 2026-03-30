"""
gru_model.py — Siamese GRU model for experiment framework.

Drop this file into experiments/models/ in the repo, then:
1. Add to experiments/models/__init__.py:
       from .gru_model import GRUModel
2. Add to MODEL_REGISTRY in experiments/run_experiment.py:
       "gru": GRUModel(),
3. Run:
       python run_experiment.py --model gru --name gru_siamese

This follows the same interface as CatBoostModel, XGBoostModel, etc.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Siamese GRU network
# ---------------------------------------------------------------------------

class _SiameseGRU(nn.Module):
    def __init__(self, embedding_dim=2560, chunk_size=256, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.seq_len = embedding_dim // chunk_size
        self.chunk_size = chunk_size

        self.gru = nn.GRU(
            input_size=chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        clf_input = 4 * (2 * hidden_size)  # [h1; h2; |h1-h2|; h1*h2]
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, 1),
        )

    def encode(self, x):
        x = x.view(-1, self.seq_len, self.chunk_size)
        _, h_n = self.gru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h

    def forward(self, emb1, emb2):
        h1 = self.encode(emb1)
        h2 = self.encode(emb2)
        combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Model wrapper 
# ---------------------------------------------------------------------------

_DEFAULTS = dict(
    embedding_dim=2560,
    chunk_size=256,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    epochs=10,
    batch_size=512,
    lr=1e-3,
    threshold=0.5,
    seed=42,
)


class GRUModel:
    name = "SiameseGRU"

    def __init__(self, **overrides):
        self.cfg = {**_DEFAULTS, **overrides}
        self.threshold = self.cfg["threshold"]
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_names = None

    # -- interface: build_features -------------------------------------------

    def build_features(self, records):
        """Return raw embedding pairs as features (no hand-crafted features).

        X is (N, 2 * embedding_dim) — emb1 concatenated with emb2.
        """
        emb1 = np.array([r.emb1 for r in records], dtype=np.float32)
        emb2 = np.array([r.emb2 for r in records], dtype=np.float32)
        y = np.array([r.label for r in records], dtype=np.int64)

        X = np.concatenate([emb1, emb2], axis=1)  # (N, 5120)
        self._feature_names = (
            [f"emb1_{i}" for i in range(emb1.shape[1])] +
            [f"emb2_{i}" for i in range(emb2.shape[1])]
        )
        return X, y, self._feature_names

    # -- interface: fit ------------------------------------------------------

    def fit(self, X_train, y_train):
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        dim = X_train.shape[1] // 2
        emb1 = torch.from_numpy(X_train[:, :dim])
        emb2 = torch.from_numpy(X_train[:, dim:])
        labels = torch.from_numpy(y_train)

        ds = TensorDataset(emb1, emb2, labels)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"],
                            shuffle=True, num_workers=4, pin_memory=True)

        self._model = _SiameseGRU(
            embedding_dim=dim,
            chunk_size=self.cfg["chunk_size"],
            hidden_size=self.cfg["hidden_size"],
            num_layers=self.cfg["num_layers"],
            dropout=self.cfg["dropout"],
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.cfg["lr"])
        criterion = nn.BCEWithLogitsLoss()

        self._model.train()
        for epoch in range(1, self.cfg["epochs"] + 1):
            total_loss = 0
            n = 0
            for e1, e2, lab in loader:
                e1 = e1.to(self._device)
                e2 = e2.to(self._device)
                lab = lab.to(self._device).float().unsqueeze(1)

                optimizer.zero_grad()
                logits = self._model(e1, e2)
                loss = criterion(logits, lab)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n += 1

            print(f"  [GRU] Epoch {epoch}/{self.cfg['epochs']}  "
                  f"loss={total_loss / n:.4f}", flush=True)

    # -- interface: predict_proba --------------------------------------------

    def predict_proba(self, X_test):
        dim = X_test.shape[1] // 2
        emb1 = torch.from_numpy(X_test[:, :dim])
        emb2 = torch.from_numpy(X_test[:, dim:])

        ds = TensorDataset(emb1, emb2)
        loader = DataLoader(ds, batch_size=self.cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

        self._model.eval()
        all_proba = []
        with torch.no_grad():
            for (e1, e2) in loader:
                e1 = e1.to(self._device)
                e2 = e2.to(self._device)
                logits = self._model(e1, e2)
                proba = torch.sigmoid(logits).cpu().numpy().flatten()
                all_proba.append(proba)

        return np.concatenate(all_proba)

    # -- interface: get_config -----------------------------------------------

    def get_config(self):
        params = sum(p.numel() for p in self._model.parameters()) if self._model else 0
        return {
            "model_class": "SiameseGRU",
            "total_params": params,
            **self.cfg,
        }
