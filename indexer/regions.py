"""Garment region extraction -- the component that makes binding real.

This is the load-bearing piece of the architecture. Everything else (soft labels,
Hungarian matching, two-stage search) is downstream of one question: WHICH PIXELS
ARE THE SHIRT? Answer that, and "red shirt + blue pants" stops being ambiguous,
because the colour of the shirt is read from the shirt.

Two interchangeable sources behind one interface (fashion.md sec.3.2):

  FashionpediaGTRegions -- ground-truth instance masks. An ORACLE: only available on
      annotated data, so it cannot be the shipped system. Its job is to upper-bound
      what perfect grounding would buy, so the detector's error can be PRICED rather
      than guessed at.

  SegformerClothesRegions -- a clothes segmenter that runs on ANY image, annotated or
      not. This is what actually ships, and what runs on the COCO supplement (which has
      no garment annotations at all).

Running both on Fashionpedia and diffing the retrieval metrics is the ablation that
tells us how much detector error costs. That number goes in the write-up.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shared.schema import is_garment

# --------------------------------------------------------------------------


class Region:
    """A garment region: a mask, its box, and whatever the source knows about it.

    `gt_category` is populated only by the oracle source. The detector leaves it None,
    and the fine-grained category is then decided by CLIP on the crop (attributes.py) --
    which is the honest path, since a deployed system has no annotations to read.
    """

    __slots__ = ("mask", "bbox", "area_frac", "gt_category", "source_label")

    def __init__(self, mask: np.ndarray, gt_category: str | None = None,
                 source_label: str | None = None):
        self.mask = mask.astype(bool)
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            self.bbox = (0, 0, 0, 0)
        else:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            self.bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        self.area_frac = float(self.mask.sum()) / self.mask.size
        self.gt_category = gt_category
        self.source_label = source_label

    def crop(self, img: Image.Image, pad: float = 0.08) -> Image.Image:
        """Crop the image to this region, with a little context padding."""
        x, y, w, h = self.bbox
        px, py = int(w * pad), int(h * pad)
        return img.crop((
            max(0, x - px), max(0, y - py),
            min(img.width, x + w + px), min(img.height, y + h + py),
        ))

    def masked_pixels(self, img: Image.Image) -> np.ndarray:
        """RGB pixels INSIDE the mask only.

        This is what makes the colour extractor work: background is excluded entirely,
        so a white shirt against a red wall reads as white, not pink. It is also why
        masks (not just boxes) are worth the compute -- a box would include the wall.
        """
        arr = np.asarray(img.convert("RGB"))
        return arr[self.mask]


class RegionSource(ABC):
    """Interface. The indexer depends on this, never on a concrete source."""

    name: str

    @abstractmethod
    def regions(self, image_id: str, img: Image.Image) -> list[Region]:
        ...


# --------------------------------------------------------------------------
# Oracle: Fashionpedia ground-truth masks
# --------------------------------------------------------------------------

class FashionpediaGTRegions(RegionSource):
    """Ground-truth instance masks. Evaluation only -- see module docstring."""

    name = "fashionpedia_gt"

    def __init__(self, ann_path: Path, min_area_frac: float = 0.005):
        from pycocotools import mask as mask_utils  # noqa: F401  (import check)

        d = json.loads(Path(ann_path).read_text())
        self.cats = {c["id"]: c for c in d["categories"]}
        self.images = {im["id"]: im for im in d["images"]}
        self.min_area_frac = min_area_frac

        # Phase 0, Finding 2: drop `sleeve`, `collar`, `pocket`, `zipper`, `bead`...
        # Indexing garment PARTS as slots would fill the index with sleeves.
        self.by_image: dict[int, list[dict]] = {}
        kept = dropped = 0
        for a in d["annotations"]:
            cat = self.cats[a["category_id"]]
            if not is_garment(cat.get("supercategory", "")):
                dropped += 1
                continue
            self.by_image.setdefault(a["image_id"], []).append(a)
            kept += 1
        print(f"[{self.name}] garment instances kept={kept} dropped(parts/closures/decor)={dropped}")

        self.fname_to_id = {im["file_name"]: i for i, im in self.images.items()}

    def regions(self, image_id: str, img: Image.Image) -> list[Region]:
        from pycocotools import mask as mask_utils

        iid = self.fname_to_id.get(image_id)
        if iid is None:
            return []

        out: list[Region] = []
        h, w = img.height, img.width
        for a in self.by_image.get(iid, []):
            seg = a.get("segmentation")
            if not seg:
                continue
            if isinstance(seg, list):                      # polygon
                rles = mask_utils.frPyObjects(seg, h, w)
                rle = mask_utils.merge(rles)
            elif isinstance(seg.get("counts"), list):      # uncompressed RLE
                rle = mask_utils.frPyObjects(seg, h, w)
            else:                                          # compressed RLE
                rle = seg
            m = mask_utils.decode(rle)
            if m.ndim == 3:
                m = m.any(axis=2)
            r = Region(m, gt_category=self.cats[a["category_id"]]["name"])
            if r.area_frac >= self.min_area_frac:
                out.append(r)
        return out


# --------------------------------------------------------------------------
# Detector: clothes segmentation, works on any image
# --------------------------------------------------------------------------

class SegformerClothesRegions(RegionSource):
    """SegFormer trained on ATR clothes parsing. This is the source that SHIPS.

    Gives a per-garment MASK on any photo, which is what the pixel colour extractor
    needs. Its label set is coarse (`Upper-clothes` covers shirt/hoodie/blazer alike) --
    that is fine and by design: the region is what we need from it, and CLIP decides the
    fine-grained category on the crop. Segmentation supplies the WHERE, CLIP the WHAT.

    Known gap: no `tie` class. Whether that hurts eval Query 5 is exactly what the
    GT-vs-detector ablation will reveal, rather than something we assume either way.
    """

    name = "segformer_clothes"

    # ATR label set of mattmdjaga/segformer_b2_clothes.
    SKIN_AND_BG = {0, 2, 11, 12, 13, 14, 15}   # background, hair, face, legs, arms
    LABELS = {
        1: "hat", 3: "glasses", 4: "upper-clothes", 5: "skirt", 6: "pants",
        7: "dress", 8: "belt", 9: "shoe", 10: "shoe", 16: "bag", 17: "scarf",
    }
    MODEL = "mattmdjaga/segformer_b2_clothes"

    def __init__(self, device: str = "cuda", min_area_frac: float = 0.005):
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        self.device = device
        self.min_area_frac = min_area_frac
        self.proc = SegformerImageProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.MODEL)
        self.model.to(device).eval()

    @torch.no_grad()
    def regions(self, image_id: str, img: Image.Image) -> list[Region]:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        up = torch.nn.functional.interpolate(
            logits, size=(img.height, img.width), mode="bilinear", align_corners=False
        )
        seg = up.argmax(dim=1)[0].cpu().numpy()

        out: list[Region] = []
        for idx, label in self.LABELS.items():
            m = seg == idx
            if not m.any():
                continue
            r = Region(m, gt_category=None, source_label=label)
            if r.area_frac >= self.min_area_frac:
                out.append(r)
        return out


# --------------------------------------------------------------------------

def build_region_source(kind: str, **kw) -> RegionSource:
    if kind == "fashionpedia_gt":
        return FashionpediaGTRegions(Path(kw["ann_path"]), kw.get("min_area_frac", 0.005))
    if kind == "segformer_clothes":
        return SegformerClothesRegions(kw.get("device", "cuda"), kw.get("min_area_frac", 0.005))
    raise ValueError(f"unknown region source: {kind!r}")
