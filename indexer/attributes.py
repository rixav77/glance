"""Per-axis attribute extraction: soft labels, pixel colour, entropy fallback.

Three ideas, each fixing a specific failure of the naive approach.

1. SOFT-LABEL EXPECTATION EMBEDDINGS (fashion.md sec.3.3)
   Do not embed the argmax label. Embed the probability-weighted MEAN of the whole
   vocabulary's label embeddings:

       e_axis = normalize( sum_v  p(v|region) * SBERT(v) )

   Taking the argmax first would collapse every image onto one of ~18 fixed vectors,
   so hundreds of images would share a byte-identical colour vector and "search" on
   that axis would be a lookup with massive ties. Keeping the distribution keeps the
   grading: a confidently-navy blazer and a navy/black coin-flip get DIFFERENT vectors,
   and the confident one ranks higher for "navy". Same storage, same cosine, no ties.

2. PIXEL COLOUR (fashion.md sec.3.4)
   Fashionpedia annotates no colours at all (Phase 0, Finding 1), so colour must be
   derived. CLIP is weak at colour -- it learns from captions, where colour words are
   sparse. A Lab-space histogram over the garment MASK is not: it reads the actual
   pixels of the actual garment, background excluded. So colour is pixel-dominant
   (lambda ~0.3 toward CLIP), with CLIP retained only because a Lab centroid cannot
   express "bright", "pastel" or "neon".

3. ENTROPY FALLBACK (fashion.md sec.3.5)
   Fire the VQA fallback on NORMALISED ENTROPY, not on a raw top-1 probability. A top
   score of 0.35 means something completely different over 8 labels than over 28 -- so
   thresholding it directly, as the original plan did, compares incomparable numbers
   across axes.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab

from indexer import vocab
from indexer.regions import Region
from shared import axis_encoding
from shared.schema import GLOBAL_AXES, SLOT_AXES


class AttributeExtractor:
    def __init__(self, clip_model, clip_proc, sbert, device="cuda",
                 color_lambda: float = 0.3, entropy_threshold: float = 0.85,
                 vqa=None, vqa_proc=None, temperature: float = 100.0,
                 fallback_axes: tuple[str, ...] = ("color",)):
        self.clip, self.clip_proc = clip_model, clip_proc
        self.sbert = sbert
        self.device = device
        self.color_lambda = color_lambda        # weight on CLIP; 1-lambda on pixels
        self.entropy_threshold = entropy_threshold
        self.vqa, self.vqa_proc = vqa, vqa_proc
        self.temperature = temperature
        # Which axes may use the VQA fallback. Smoke test showed VQA HELPS on colour
        # (out-of-vocab shades) but HURTS on category (segmenter+constrained CLIP is
        # already better) and pattern (VQA returns junk). So: colour only, by default.
        self.fallback_axes = set(fallback_axes)

        self._text_emb: dict[str, torch.Tensor] = {}    # CLIP text emb per axis vocab
        self._label_emb: dict[str, np.ndarray] = {}     # SBERT emb per axis vocab
        self._prep_vocab()
        self._prep_color_anchors()

    # -- vocabulary precomputation ----------------------------------------
    @torch.no_grad()
    def _prep_vocab(self) -> None:
        for axis, mapping in vocab.AXIS_VOCAB.items():
            prompts = list(mapping.values())
            t = self.clip_proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            e = self.clip.get_text_features(**t)
            self._text_emb[axis] = e / e.norm(dim=-1, keepdim=True)
            # SBERT embeddings of the bare LABELS (not the prompts) -- the query side
            # embeds a bare phrase too, so both sides must live in the same space.
            self._label_emb[axis] = self.sbert.encode(
                list(mapping), normalize_embeddings=True
            )

    def _prep_color_anchors(self) -> None:
        names = list(vocab.COLOR_RGB)
        rgb = np.array([vocab.COLOR_RGB[n] for n in names], dtype=np.float64) / 255.0
        self._color_names = names
        self._color_lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    # -- core -------------------------------------------------------------
    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> torch.Tensor:
        """Encode once. Every axis then reuses this -- CLIP is the expensive part,
        and the image does not change between axes."""
        px = self.clip_proc(images=[img], return_tensors="pt").to(self.device)
        e = self.clip.get_image_features(**px)
        return e / e.norm(dim=-1, keepdim=True)

    def _dist(self, axis: str, emb: torch.Tensor) -> np.ndarray:
        logits = self.temperature * (emb @ self._text_emb[axis].T)
        return logits.softmax(dim=-1)[0].cpu().numpy()

    def _pixel_color_dist(self, region: Region, img: Image.Image) -> np.ndarray:
        """Colour distribution from the garment's own pixels, in CIELAB.

        Lab rather than RGB because Euclidean distance in Lab tracks perceived colour
        difference; in RGB it does not, so an RGB nearest-centroid would call things
        "green" that no human would.
        """
        px = region.masked_pixels(img)
        if len(px) < 30:
            return np.ones(len(self._color_names)) / len(self._color_names)

        if len(px) > 5000:                       # subsample; 5k pixels is plenty
            idx = np.random.default_rng(0).choice(len(px), 5000, replace=False)
            px = px[idx]

        lab = rgb2lab((px / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
        d = np.linalg.norm(lab[:, None, :] - self._color_lab[None, :, :], axis=2)
        nearest = d.argmin(axis=1)               # per-pixel nearest colour name
        counts = np.bincount(nearest, minlength=len(self._color_names)).astype(float)

        dist = counts / counts.sum()
        # Order this vector to match the axis vocabulary, which may differ from COLOR_RGB.
        out = np.zeros(len(vocab.COLOR))
        for i, name in enumerate(vocab.COLOR):
            if name in self._color_names:
                out[i] = dist[self._color_names.index(name)]
        s = out.sum()
        return out / s if s > 0 else np.ones(len(vocab.COLOR)) / len(vocab.COLOR)

    @staticmethod
    def _norm_entropy(p: np.ndarray) -> float:
        """0.0 = fully confident, 1.0 = uniform. Comparable ACROSS vocab sizes."""
        p = p[p > 0]
        if len(p) <= 1:
            return 0.0
        return float(-(p * np.log(p)).sum() / np.log(len(p)))

    def _soft_embedding(self, axis: str, dist: np.ndarray) -> np.ndarray:
        e = dist @ self._label_emb[axis]         # probability-weighted mean
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    @torch.no_grad()
    def _vqa(self, axis: str, img: Image.Image) -> str | None:
        """BLIP-VQA on the REGION CROP -- so the answer stays bound to the garment."""
        if self.vqa is None:
            return None
        q = vocab.VQA_QUESTION[axis]
        inputs = self.vqa_proc(img, q, return_tensors="pt").to(self.device)
        out = self.vqa.generate(**inputs, max_new_tokens=8)
        ans = self.vqa_proc.decode(out[0], skip_special_tokens=True).strip()
        return ans or None

    # -- public -----------------------------------------------------------
    def slot_attributes(self, region: Region, img: Image.Image) -> dict:
        """Extract category / colour / pattern FROM THIS REGION'S PIXELS."""
        crop = region.crop(img)
        crop_emb = self.embed_image(crop)          # encode the crop once, reuse per axis
        dists, labels, embs, fallbacks = {}, {}, {}, []

        # Slot axes are stored as their vocabulary DISTRIBUTION (unit-normed), not an
        # SBERT embedding -- see shared/axis_encoding: SBERT's colour geometry is too
        # flat to bind colour, whereas cos(query_onehot, dist) = probability mass on the
        # queried value, which separates red from white sharply.

        # -- category: GT hard label > segmenter-constrained CLIP > free CLIP --
        if region.gt_category is not None:
            labels["category"] = region.gt_category
            cdist = axis_encoding.soft_vocab_vector("category", region.gt_category, self.sbert)
            embs["category"] = cdist
            dists["category"] = cdist
        else:
            cdist = self._dist("category", crop_emb)
            allowed = vocab.allowed_category_indices(region.source_label)
            if allowed is not None:
                mask = np.zeros_like(cdist)
                mask[allowed] = 1.0
                cdist = cdist * mask
                cdist = cdist / cdist.sum() if cdist.sum() > 0 else mask / mask.sum()
            labels["category"] = vocab.labels("category")[int(cdist.argmax())]
            embs["category"] = axis_encoding.dist_vector(cdist)
            dists["category"] = cdist

        # -- colour and pattern -------------------------------------------
        for axis in ("color", "pattern"):
            dist = self._dist(axis, crop_emb)
            if axis == "color":
                pixel = self._pixel_color_dist(region, img)
                dist = self.color_lambda * dist + (1 - self.color_lambda) * pixel

            names = vocab.labels(axis)
            ent = self._norm_entropy(dist)

            if (axis in self.fallback_axes and self.vqa is not None
                    and ent > self.entropy_threshold):
                # Distribution uninformative on an axis where VQA actually helps -- ask.
                # Fold the free-text answer back onto the vocabulary (soft-mapped), so it
                # lives in the same distribution space as everything else on this axis.
                ans = self._vqa(axis, crop)
                if ans:
                    fallbacks.append(axis)
                    labels[axis] = ans
                    qv = axis_encoding.soft_vocab_vector(axis, ans, self.sbert)
                    embs[axis] = qv
                    dists[axis] = dist
                    continue

            labels[axis] = names[int(dist.argmax())]
            embs[axis] = axis_encoding.dist_vector(dist)
            dists[axis] = dist

        return {"labels": labels, "embeddings": embs, "dists": dists,
                "fallback_axes": fallbacks}

    def global_attributes(self, img: Image.Image, img_emb: torch.Tensor | None = None) -> dict:
        """Scene and style_vibe -- genuinely image-level, no region to bind to.

        Also returns the raw CLIP image embedding as `visual_global`: the fine detail
        (exact shade, texture) that discretising onto a label vocabulary throws away.
        It is a small tie-breaking term in the score, never a primary signal.
        """
        emb = self.embed_image(img) if img_emb is None else img_emb
        labels, embs, dists = {}, {}, {}
        for axis in GLOBAL_AXES:
            dist = self._dist(axis, emb)
            names = vocab.labels(axis)
            labels[axis] = names[int(dist.argmax())]
            embs[axis] = self._soft_embedding(axis, dist)
            dists[axis] = dist
        return {"labels": labels, "embeddings": embs, "dists": dists,
                "visual_global": emb[0].cpu().numpy()}
