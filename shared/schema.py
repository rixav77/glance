"""The shared contract between the indexer and the retriever.

This is the single source of truth for what an axis IS. Both pipelines import from
here, so they cannot silently drift apart -- which is the modularity the assignment
asks about ("is your logic separated from your data?").

The central design decision (fashion.md sec.3.1) lives here as a type distinction:

  SLOT axes   (category, color, pattern)  belong to ONE GARMENT REGION.
  GLOBAL axes (scene, style_vibe)         belong to THE WHOLE IMAGE.

Color and pattern always modify a garment, so they are never stored image-level --
that is exactly the mistake that lets "red shirt + blue pants" and "blue shirt + red
pants" collapse to the same index entry. Scene and vibe have no region to bind to, so
they stay global. Nothing else in the system needs to know why; it just reads this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------

SLOT_AXES = ("category", "color", "pattern")
GLOBAL_AXES = ("scene", "style_vibe")
ALL_AXES = SLOT_AXES + GLOBAL_AXES

AxisName = Literal["category", "color", "pattern", "scene", "style_vibe"]

# Qdrant collection names.
SLOT_COLLECTION = "glance_slots"
IMAGE_COLLECTION = "glance_images"

# Vector dims. MiniLM is 384; CLIP ViT-B/32 is 512.
SBERT_DIM = 384
CLIP_DIM = 512


# --------------------------------------------------------------------------
# Index-side records
# --------------------------------------------------------------------------

@dataclass
class Slot:
    """One garment region, with its attributes bound to it by construction."""

    image_id: str
    slot_id: str
    bbox: tuple[int, int, int, int]          # x, y, w, h
    area_frac: float                         # region area / image area

    # Per-axis soft-label embedding (fashion.md sec.3.3): the probability-weighted
    # mean of the axis vocabulary's label embeddings. Graded, not argmax -- so a
    # confidently-navy blazer and an ambiguous one do NOT get identical vectors.
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    # Human-readable argmax label per axis, for debugging and contact sheets.
    labels: dict[str, str] = field(default_factory=dict)
    # Full distribution per axis, kept for the entropy-based fallback and analysis.
    dists: dict[str, np.ndarray] = field(default_factory=dict)
    # Which axes fell back to VQA because the distribution was uninformative.
    fallback_axes: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "image_id": self.image_id,
            "slot_id": self.slot_id,
            "bbox": list(self.bbox),
            "area_frac": round(self.area_frac, 4),
            "labels": self.labels,
            "fallback_axes": self.fallback_axes,
        }


@dataclass
class ImageRecord:
    """One image: its global axes, plus pointers to its slots."""

    image_id: str
    image_path: str
    source: str                              # "fashionpedia" | "coco_supplement"
    slot_ids: list[str] = field(default_factory=list)

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)   # scene, style_vibe
    visual_global: np.ndarray | None = None                           # raw CLIP image emb
    labels: dict[str, str] = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "source": self.source,
            "slot_ids": self.slot_ids,
            "labels": self.labels,
        }


# --------------------------------------------------------------------------
# Query-side records
# --------------------------------------------------------------------------

@dataclass
class GarmentConstraint:
    """One garment the query asks for, with ITS OWN attributes.

    A list of these is what makes "a red tie and a white shirt" expressible at all.
    The flat one-slot-per-axis schema of the original plan could not represent it,
    which is why that plan had to declare Query 5 a known limitation.
    """

    category: str | None = None
    color: str | None = None
    pattern: str | None = None

    def specified(self) -> dict[str, str]:
        """Only the axes the query actually mentioned. Absent axes get weight 0."""
        return {a: v for a in SLOT_AXES if (v := getattr(self, a)) is not None}

    def is_empty(self) -> bool:
        return not self.specified()


@dataclass
class ParsedQuery:
    """A natural-language query, decomposed onto the same axes as the index."""

    raw: str
    garments: list[GarmentConstraint] = field(default_factory=list)
    scene: str | None = None
    style_vibe: str | None = None

    def global_specified(self) -> dict[str, str]:
        return {a: v for a in GLOBAL_AXES if (v := getattr(self, a)) is not None}

    def is_empty(self) -> bool:
        return not self.garments and not self.global_specified()


# --------------------------------------------------------------------------
# Fashionpedia category filtering
# --------------------------------------------------------------------------

# Phase 0, Finding 2: Fashionpedia's 46 categories mix real garments with garment
# PARTS -- `sleeve` is the second most common annotation in the whole dataset, and
# `collar`, `pocket`, `neckline`, `zipper`, `bead` are all annotated instances.
# Indexing those as slots would fill the index with sleeves and quietly wreck the
# assignment step in the retriever. Keep only supercategories that are garments.
GARMENT_SUPERCATEGORIES = frozenset({
    "upperbody", "lowerbody", "wholebody", "neck", "waist",
    "legs and feet", "head", "arms and hands", "others",
})
NON_GARMENT_SUPERCATEGORIES = frozenset({
    "garment parts", "closures", "decorations",
})


def is_garment(supercategory: str) -> bool:
    return supercategory in GARMENT_SUPERCATEGORIES
