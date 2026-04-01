"""
models/ensemble_model.py — Ensemble / stacking wrapper.

Supports two combination strategies:
  - 'mean'     : simple (optionally weighted) average of base-model probabilities
  - 'stacking' : trains a lightweight LogReg meta-learner on out-of-fold
                 probability vectors produced by the base models.

Design constraints
------------------
Each base model owns its own feature space (XGBoost uses matryoshka stats,
GRU v3 uses raw embeddings, etc.), so a shared feature matrix is impossible.
Instead, EnsembleModel:

  1. build_features()  — calls every member's build_features() and stores the
                         full per-member X arrays internally.  Returns a thin
                         (N, n_members) stub so run_experiment.py can slice
                         train/test rows as usual without any code changes.

  2. fit()             — uses the stub's row-count to recover the train portion
                         of each member's stored X, then fits each member on
                         its own feature space.  For stacking a meta-learner is
                         also trained on OOF predictions.

  3. predict_proba()   — recovers each member's test rows (everything after
                         n_train in the stored full arrays), collects their
                         probability vectors, and combines them.

This means run_experiment.py, report.py, and the split logic all remain
entirely unchanged.

Adding to run_experiment.py
---------------------------
  from models import EnsembleModel, XGBoostModel, CatBoostModel, GRUModelV3

  MODEL_REGISTRY = {
      ...
      "ensemble_mean":  EnsembleModel(
                            members=[XGBoostModel(), CatBoostModel(), GRUModelV3()],
                            strategy="mean",
                        ),
      "ensemble_stack": EnsembleModel(
                            members=[XGBoostModel(), CatBoostModel(), GRUModelV3()],
                            strategy="stacking",
                        ),
  }

Then run as normal:
  python run_experiment.py --model ensemble_mean --name my_ensemble_run
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import PairRecord


class EnsembleModel:
    """
    Ensemble wrapper that satisfies the standard model contract:
        build_features(records)  → (X_stub, y, feature_names)
        fit(X_stub, y)
        predict_proba(X_stub)    → 1-D float32 probability array

    Parameters
    ----------
    members  : list of model instances, each with build_features / fit /
               predict_proba.  Members can have completely different feature
               spaces — each is evaluated independently.
    strategy : "mean"     — (optionally weighted) average of probabilities.
               "stacking" — trains a LogReg meta-learner on OOF predictions,
                            then re-fits every base model on the full training
                            set before inference.
    weights  : Optional list of floats (length == len(members)) used when
               strategy="mean".  Normalised internally, so [2, 1, 1] is fine.
               Ignored when strategy="stacking".
    meta_folds : Number of CV folds for the OOF stacking phase (default 5).
    threshold  : Decision threshold exposed to run_experiment.py (default 0.5).
    """

    name = "Ensemble"

    def __init__(
        self,
        members: list,
        strategy: str = "mean",
        weights: list[float] | None = None,
        meta_folds: int = 5,
        threshold: float = 0.5,
    ):
        if strategy not in ("mean", "stacking"):
            raise ValueError(f"strategy must be 'mean' or 'stacking', got {strategy!r}")
        if weights is not None and len(weights) != len(members):
            raise ValueError("len(weights) must equal len(members)")

        self.members    = members
        self.strategy   = strategy
        self.weights    = weights
        self.meta_folds = meta_folds
        self.threshold  = threshold

        # Full per-member feature arrays (set in build_features, sliced in fit/predict)
        self._member_X_all: list[np.ndarray] = []
        self._n_train: int = 0   # set during fit(); used by predict_proba()

        # Stacking components
        self._meta_scaler = StandardScaler()
        self._meta_clf    = LogisticRegression(max_iter=1000, random_state=42)

        # Feature names reported to run_experiment.py / report.py
        self._feature_names: list[str] = [
            f"base_{getattr(m, 'name', type(m).__name__)}_proba"
            for m in members
        ]

    # ------------------------------------------------------------------ #
    # build_features                                                       #
    # ------------------------------------------------------------------ #

    def build_features(
        self, records: list[PairRecord]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Call each member's build_features() independently and store their
        full feature matrices.  Return a thin stub matrix so that
        run_experiment.py can slice train/test rows without modification.

        The stub shape is (N, n_members) — one column per base model.
        It is never actually used for learning; it is only a vessel for
        the row-count that fit() and predict_proba() need.
        """
        self._member_X_all = []
        ys: list[np.ndarray] = []

        for i, model in enumerate(self.members):
            mname = getattr(model, "name", type(model).__name__)
            print(f"  [Ensemble] build_features: member {i+1}/{len(self.members)} — {mname}", flush=True)
            X_m, y_m, _ = model.build_features(records)
            self._member_X_all.append(X_m)
            ys.append(y_m)

        # Sanity: all members must produce the same labels
        for j, y_m in enumerate(ys[1:], start=1):
            if not np.array_equal(ys[0], y_m):
                raise RuntimeError(
                    f"Member {j} produced different labels than member 0. "
                    "All members must operate on the same records."
                )

        y = ys[0]
        n = len(records)
        stub = np.zeros((n, len(self.members)), dtype=np.float32)
        return stub, y, self._feature_names

    # ------------------------------------------------------------------ #
    # fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X_train_stub: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit every base model on its own feature matrix.

        The stub's row count tells us how many rows belong to the training
        split, so we can recover each member's X_train from _member_X_all.
        """
        self._n_train = len(X_train_stub)
        member_X_trains = [X_m[: self._n_train] for X_m in self._member_X_all]

        if self.strategy == "stacking":
            self._fit_stacking(member_X_trains, y_train)
        else:
            self._fit_mean(member_X_trains, y_train)

    def _fit_mean(
        self,
        member_X_trains: list[np.ndarray],
        y_train: np.ndarray,
    ) -> None:
        """Fit each base model independently on its training slice."""
        for i, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
            mname = getattr(model, "name", type(model).__name__)
            print(f"  [Ensemble] Fitting member {i+1}/{len(self.members)} — {mname}", flush=True)
            model.fit(X_m, y_train)

    def _fit_stacking(
        self,
        member_X_trains: list[np.ndarray],
        y_train: np.ndarray,
    ) -> None:
        """
        1. Collect OOF predictions from each base model using k-fold CV.
        2. Train a LogReg meta-learner on the OOF probability matrix.
        3. Re-fit every base model on the full training set for inference.
        """
        n = len(y_train)
        oof = np.zeros((n, len(self.members)), dtype=np.float32)

        skf = StratifiedKFold(
            n_splits=self.meta_folds, shuffle=True, random_state=42
        )

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(np.arange(n), y_train)):
            print(
                f"  [Ensemble] Stacking fold {fold_i+1}/{self.meta_folds} "
                f"(train={len(tr_idx)}, val={len(val_idx)})",
                flush=True,
            )
            for j, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
                mname = getattr(model, "name", type(model).__name__)
                print(
                    f"    [Ensemble]   member {j+1}/{len(self.members)} — {mname}",
                    flush=True,
                )
                model.fit(X_m[tr_idx], y_train[tr_idx])
                oof[val_idx, j] = model.predict_proba(X_m[val_idx])

        # Train meta-learner
        print("  [Ensemble] Training meta-learner on OOF predictions...", flush=True)
        oof_scaled = self._meta_scaler.fit_transform(oof)
        self._meta_clf.fit(oof_scaled, y_train)
        print(
            f"  [Ensemble] Meta-learner coefficients: {self._meta_clf.coef_.round(3)}",
            flush=True,
        )

        # Re-fit every base model on the *full* training set
        print("  [Ensemble] Re-fitting base models on full training set...", flush=True)
        for i, (model, X_m) in enumerate(zip(self.members, member_X_trains)):
            mname = getattr(model, "name", type(model).__name__)
            print(
                f"  [Ensemble] Final fit — member {i+1}/{len(self.members)} — {mname}",
                flush=True,
            )
            model.fit(X_m, y_train)

    # ------------------------------------------------------------------ #
    # predict_proba                                                        #
    # ------------------------------------------------------------------ #

    def predict_proba(self, X_test_stub: np.ndarray) -> np.ndarray:
        """
        Recover each member's test rows from the stored full arrays, collect
        probability vectors, then combine via the chosen strategy.
        """
        # Member test rows are everything after n_train in the stored full arrays
        member_X_tests = [X_m[self._n_train :] for X_m in self._member_X_all]

        probas = np.stack(
            [
                model.predict_proba(X_t)
                for model, X_t in zip(self.members, member_X_tests)
            ],
            axis=1,
        )  # (N_test, n_members)

        if self.strategy == "stacking":
            return (
                self._meta_clf.predict_proba(
                    self._meta_scaler.transform(probas)
                )[:, 1]
                .astype(np.float32)
            )

        # "mean" — optionally weighted
        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float32)
        else:
            w = np.ones(len(self.members), dtype=np.float32)
        w = w / w.sum()
        return (probas * w).sum(axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Optional extras consumed by report.py                               #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        member_configs = []
        for m in self.members:
            if hasattr(m, "get_config"):
                try:
                    member_configs.append(m.get_config())
                except Exception as exc:
                    member_configs.append({"error": str(exc)})
            else:
                member_configs.append({
                    "model_class": getattr(m, "name", type(m).__name__)
                })

        return {
            "model_class": "EnsembleModel",
            "strategy":    self.strategy,
            "weights":     self.weights,
            "meta_folds":  self.meta_folds if self.strategy == "stacking" else None,
            "threshold":   self.threshold,
            "n_members":   len(self.members),
            "members":     [getattr(m, "name", type(m).__name__) for m in self.members],
            "member_configs": member_configs,
        }
