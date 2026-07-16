"""Vocabulary-distribution encoding for the discrete slot axes (category, colour, pattern).

Why this exists (the smoke-test fix). Routing colour through SBERT made "red" and
"white" sit 0.63 apart and "red"/"blue" 0.73 apart -- nearly as close as a true match --
so colour could not bind. The cure is to stop using a semantic text space for discrete
attributes and instead represent each slot axis as its PROBABILITY DISTRIBUTION over the
axis vocabulary, and each query value as a (soft) one-hot over the same vocabulary.

Then cos(query_onehot, slot_dist / ||slot_dist||) is proportional to the probability the
slot assigns to the queried value -- i.e. "how red is this garment" -- which separates
red from white sharply. Out-of-vocabulary query words (e.g. "crimson") soft-map to their
nearest vocabulary neighbours via SBERT, so generality is preserved, not lost.

Both pipelines import this, so the index side and the query side build vectors the same
way -- keeping the "logic separated from data" contract intact.
"""

from __future__ import annotations

import numpy as np

from indexer import vocab


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


_LABEL_EMB_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _label_emb(axis: str, sbert) -> np.ndarray:
    """SBERT embeddings of an axis's labels, computed once and cached. Encoding the
    vocabulary on every query value made the swap test unrunnable on CPU."""
    key = (id(sbert), axis)
    e = _LABEL_EMB_CACHE.get(key)
    if e is None:
        e = sbert.encode(vocab.labels(axis), normalize_embeddings=True)
        _LABEL_EMB_CACHE[key] = e
    return e


def onehot(axis: str, label: str) -> np.ndarray | None:
    labels = vocab.labels(axis)
    if label in labels:
        v = np.zeros(len(labels))
        v[labels.index(label)] = 1.0
        return v
    return None


def soft_vocab_vector(axis: str, value: str, sbert, temperature: float = 0.07) -> np.ndarray:
    """Query value -> unit vector over `axis` vocabulary.

    Exact vocabulary hit -> one-hot (the overwhelmingly common case). Otherwise a sharp
    softmax over SBERT similarity to the vocabulary labels, so an unseen word lands on
    its nearest known neighbours instead of failing.
    """
    oh = onehot(axis, value)
    if oh is not None:
        return oh                      # already unit norm

    lab_emb = _label_emb(axis, sbert)
    q = sbert.encode([value], normalize_embeddings=True)[0]
    sims = lab_emb @ q
    w = np.exp((sims - sims.max()) / temperature)
    w = w / w.sum()
    return l2(w)


def semantic_vocab_vector(axis: str, value: str, sbert, temperature: float = 0.10) -> np.ndarray:
    """Query value -> unit vector over `axis` vocabulary, ALWAYS soft.

    Unlike soft_vocab_vector (one-hot on an exact hit), this spreads mass to semantic
    neighbours even for in-vocabulary words -- so a "shirt" query partially matches a
    "blouse" or "top" slot. Used for CATEGORY, where near-synonyms should match; colour
    and pattern stay exact (soft_vocab_vector), because "red" must NOT match "orange".
    """
    lab_emb = _label_emb(axis, sbert)
    q = sbert.encode([value], normalize_embeddings=True)[0]
    sims = lab_emb @ q
    w = np.exp((sims - sims.max()) / temperature)
    w = w / w.sum()
    return l2(w)


def dist_vector(dist: np.ndarray) -> np.ndarray:
    """Slot side: store the axis probability distribution as a unit vector for cosine."""
    return l2(np.asarray(dist, dtype=np.float64))


def slot_axis_dims() -> dict[str, int]:
    """Vector dim per slot axis = size of that axis vocabulary."""
    from shared.schema import SLOT_AXES
    return {a: len(vocab.labels(a)) for a in SLOT_AXES}
