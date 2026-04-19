"""
featurizers/char_ngram.py — Train-fitted character n-gram featurizer for question pairs.

This featurizer must be fit on training questions only before being used to
transform PairRecords.  It computes similarity features derived from
bag-of-character-n-gram representations (n = 1 … 8).

Two variants are produced:
  • TF-IDF reweighted (analyzer='char_wb', sublinear_tf=True)
  • Binary occurrence (plain presence/absence, no TF-IDF weighting)

Feature groups
--------------
(A) TF-IDF character n-gram similarity
    char_tfidf_cosine_sim  — cosine similarity of the two TF-IDF vectors
    char_tfidf_l1_diff     — L1  norm of |tfidf(q1) - tfidf(q2)|
    char_tfidf_l2_diff     — L2  norm of |tfidf(q1) - tfidf(q2)|
    char_tfidf_dot         — dot product of raw (un-normalised) TF-IDF vectors

(B) Binary (unweighted) character n-gram overlap
    char_bin_cosine_sim    — cosine similarity of binary indicator vectors
    char_bin_jaccard       — |support(q1) ∩ support(q2)| / |support(q1) ∪ support(q2)|
                             ("support" = the set of n-grams present in the question)

Performance / memory note
-------------------------
Each question vector is stored as a **sparse CSR row** plus the integer set of
its non-zero column indices (used for fast binary Jaccard / cosine).  With
``max_features = 100 000`` and ~450 k unique training questions, keeping sparse
rows reduces memory from ~180 GB (dense float32) to a few hundred MB.

Vectors are batch-computed during ``fit()`` / ``cache_questions()`` and cached
keyed by question string.  ``transform()`` is therefore O(nnz) per pair for
seen questions.  Unseen questions are computed and cached on demand.

Usage
-----
    from featurizers import CharNgramFeaturizer

    featurizer = CharNgramFeaturizer()
    train_qs = [r.question1 for r in train] + [r.question2 for r in train]
    featurizer.fit(train_qs)
    featurizer.cache_questions([r.question1 for r in test] + [r.question2 for r in test])

    feats = featurizer.transform(pair_record)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from data import PairRecord


_LOG_PREFIX = "[CharNgramFeaturizer]"

# Number of questions per batch when pre-computing vectors.  Keeps peak
# transient memory bounded, which matters for big char-ngram vocabularies.
_CACHE_BATCH_SIZE: int = 50_000


def _fmt_secs(seconds: float) -> str:
    """Format seconds as a short human-readable string."""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.2f}s"


class CharNgramFeaturizer:
    """
    Train-fitted featurizer based on bag-of-character-n-grams (1 ≤ n ≤ 8).

    Parameters
    ----------
    ngram_range : tuple[int, int]
        (min_n, max_n) for character n-grams.  Default (1, 8).
    max_features : int | None
        Vocabulary cap for each internal TfidfVectorizer.
    sublinear_tf : bool
        Apply sublinear (log) TF scaling in the TF-IDF vectorizer.
    analyzer : str
        'char_wb' (pads at word boundaries, default) or 'char'.
    verbose : bool
        If True (default), print progress logs during fit / caching.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 8),
        max_features: int | None = 100_000,
        sublinear_tf: bool = True,
        analyzer: str = "char_wb",
        *,
        verbose: bool = True,
    ) -> None:
        self._ngram_range = ngram_range
        self._max_features = max_features
        self._sublinear_tf = sublinear_tf
        self._analyzer = analyzer
        self._verbose = verbose

        self._tfidf_vec: TfidfVectorizer | None = None
        self._fitted: bool = False

        # cache: question → (raw_tfidf_csr_row, normed_tfidf_csr_row, support_indices)
        #   raw / normed  : scipy.sparse.csr_matrix of shape (1, vocab_size)
        #   support       : np.ndarray of int column indices where the vector
        #                   is non-zero, used for O(nnz) Jaccard/binary-cosine.
        self._cache: dict[
            str, tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]
        ] = {}

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"{_LOG_PREFIX} {msg}", flush=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, questions: list[str]) -> "CharNgramFeaturizer":
        """
        Fit on a flat list of question strings (training data only).
        All unique questions are automatically cached after fitting.
        """
        t0 = time.time()
        n_docs = len(questions)
        n_unique = len(set(questions))
        self._log(
            f"fit(): starting char-ngram TF-IDF fit on {n_docs:,} questions "
            f"({n_unique:,} unique) | "
            f"analyzer={self._analyzer!r}, ngram_range={self._ngram_range}, "
            f"max_features={self._max_features}, sublinear_tf={self._sublinear_tf}"
        )

        self._tfidf_vec = TfidfVectorizer(
            analyzer=self._analyzer,
            ngram_range=self._ngram_range,
            max_features=self._max_features,
            sublinear_tf=self._sublinear_tf,
            smooth_idf=True,
        )
        self._tfidf_vec.fit(questions)
        self._fitted = True

        fit_elapsed = time.time() - t0
        self._log(
            f"fit(): TF-IDF fit complete in {_fmt_secs(fit_elapsed)} | "
            f"vocab_size={len(self._tfidf_vec.vocabulary_):,}"
        )

        self._log(
            f"fit(): pre-caching vectors for {n_unique:,} unique training questions …"
        )
        self.cache_questions(questions)

        total_elapsed = time.time() - t0
        self._log(
            f"fit(): done in {_fmt_secs(total_elapsed)} "
            f"(cache size={len(self._cache):,} vectors)"
        )
        return self

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache_questions(self, questions: list[str]) -> None:
        """
        Pre-compute and cache vectors for a list of questions.
        Safe to call multiple times; already-cached strings are skipped.

        Vectors are stored as sparse CSR rows (raw + L2-normed) plus the
        integer support (column indices) of each row.  This keeps memory
        proportional to the total number of non-zero entries (~ O(total chars))
        rather than O(n_questions × vocab_size), which would be ~180 GB at
        the defaults.
        """
        self._check_fitted()
        assert self._tfidf_vec is not None

        t0 = time.time()
        n_requested = len(questions)
        unique = [q for q in dict.fromkeys(questions) if q not in self._cache]
        n_new = len(unique)
        n_cached_hits = n_requested - n_new if n_requested >= n_new else 0

        if not unique:
            self._log(
                f"cache_questions(): nothing to do "
                f"({n_requested:,} requested, all already cached)"
            )
            return

        self._log(
            f"cache_questions(): transforming {n_new:,} new questions "
            f"({n_requested:,} requested, {n_cached_hits:,} already in cache) | "
            f"vocab_size={len(self._tfidf_vec.vocabulary_):,} (sparse storage)"
        )

        # Batch to bound peak memory during .transform()
        for start in range(0, n_new, _CACHE_BATCH_SIZE):
            end = min(start + _CACHE_BATCH_SIZE, n_new)
            chunk = unique[start:end]

            sparse = self._tfidf_vec.transform(chunk).tocsr()                 # (b, vocab)
            sparse = sparse.astype(np.float32, copy=False)
            normed = normalize(sparse, norm="l2", axis=1, copy=True)           # row-wise L2

            # Pre-extract per-row CSR slices and the non-zero column indices.
            # We build the support arrays from the parent CSR's indptr/indices
            # buffers, which is much faster than re-materialising per row.
            indptr = sparse.indptr
            indices = sparse.indices
            for j, q in enumerate(chunk):
                row_start, row_end = indptr[j], indptr[j + 1]
                support = indices[row_start:row_end].copy()
                self._cache[q] = (
                    sparse.getrow(j),
                    normed.getrow(j),
                    support,
                )

            if self._verbose and end < n_new:
                self._log(
                    f"cache_questions(): progress {end:,}/{n_new:,} "
                    f"({end / n_new:.0%}) in {_fmt_secs(time.time() - t0)}"
                )

        elapsed = time.time() - t0
        rate = n_new / elapsed if elapsed > 0 else float("inf")
        self._log(
            f"cache_questions(): cached {n_new:,} vectors in {_fmt_secs(elapsed)} "
            f"({rate:,.0f} q/s) | total cache size={len(self._cache):,}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CharNgramFeaturizer is not fitted. "
                "Call .fit(questions) with training questions first."
            )

    def _get_vectors(
        self, text: str
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """Return (raw_csr_row, normed_csr_row, support_indices) using cache."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        # on-demand for unseen strings
        self.cache_questions([text])
        return self._cache[text]

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, r: "PairRecord") -> dict[str, float]:
        """
        Compute character n-gram similarity features for one pair.

        Returns
        -------
        dict[str, float]
            char_tfidf_cosine_sim, char_tfidf_l1_diff, char_tfidf_l2_diff,
            char_tfidf_dot,
            char_bin_cosine_sim, char_bin_jaccard
        """
        self._check_fitted()

        raw1, norm1, sup1 = self._get_vectors(r.question1)
        raw2, norm2, sup2 = self._get_vectors(r.question2)

        # (A) TF-IDF reweighted features — all operations on sparse rows.
        diff = (raw1 - raw2).tocsr()
        diff_data = diff.data
        if diff_data.size > 0:
            abs_data = np.abs(diff_data)
            char_tfidf_l1_diff = float(abs_data.sum())
            char_tfidf_l2_diff = float(np.sqrt((abs_data * abs_data).sum()))
        else:
            char_tfidf_l1_diff = 0.0
            char_tfidf_l2_diff = 0.0

        char_tfidf_cosine_sim = float(norm1.multiply(norm2).sum())
        char_tfidf_dot        = float(raw1.multiply(raw2).sum())

        # (B) Binary overlap features — use the integer support sets.
        #
        # |intersection| = number of n-grams present in BOTH questions
        # |union|        = number of n-grams present in EITHER question
        # Cosine of 0/1 vectors = |∩| / sqrt(|sup1| * |sup2|)
        set1 = sup1
        set2 = sup2
        n1 = int(set1.size)
        n2 = int(set2.size)

        if n1 == 0 or n2 == 0:
            inter = 0
        else:
            # np.intersect1d on pre-sorted (scipy's indices are sorted) ints
            # is O(n1 + n2) with assume_unique=True.
            inter = int(np.intersect1d(set1, set2, assume_unique=True).size)
        union_size = n1 + n2 - inter

        bin_cos_den = np.sqrt(n1 * n2)
        char_bin_cosine_sim = float(inter / bin_cos_den) if bin_cos_den > 0 else 0.0
        char_bin_jaccard    = float(inter / union_size) if union_size > 0 else 0.0

        return {
            "char_tfidf_cosine_sim": char_tfidf_cosine_sim,
            "char_tfidf_l1_diff":    char_tfidf_l1_diff,
            "char_tfidf_l2_diff":    char_tfidf_l2_diff,
            "char_tfidf_dot":        char_tfidf_dot,
            "char_bin_cosine_sim":   char_bin_cosine_sim,
            "char_bin_jaccard":      char_bin_jaccard,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"fitted, vocab_size={len(self._tfidf_vec.vocabulary_)}" \
            if self._fitted and self._tfidf_vec is not None else "not fitted"
        return (
            f"CharNgramFeaturizer("
            f"ngram_range={self._ngram_range}, "
            f"max_features={self._max_features}, "
            f"analyzer={self._analyzer!r}, "
            f"{status})"
        )
