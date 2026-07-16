"""Two-stage retrieval with structural binding (fashion.md sec.4.2-4.3).

Stage 1 -- candidate generation (ANN, sub-linear). Each query garment ANN-searches the
   `slots` collection on its most selective axis; globals search `images`. Union the
   image_ids. A fixed handful of HNSW lookups regardless of corpus size -- this is what
   scales to 1M images.

Stage 2 -- exact rerank with binding (O(candidates)). For each candidate image, solve
   the assignment between query garments and image slots with the Hungarian algorithm,
   so each colour is scored against the pixels of ITS OWN garment. This is the step that
   makes "red tie + white shirt" beat "white tie + red shirt": the binding is structural,
   not a bag of words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from scipy.optimize import linear_sum_assignment

from shared import axis_encoding
from shared.schema import GLOBAL_AXES, ParsedQuery


@dataclass
class SearchResult:
    image_id: str
    image_path: str
    score: float
    source: str
    matches: list[dict] = field(default_factory=list)   # per query-garment binding detail
    scene_label: str | None = None
    breakdown: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, qdrant_path: str, sbert, clip_model=None, clip_proc=None,
                 slot_collection="glance_slots", image_collection="glance_images",
                 weights: dict | None = None, device="cuda"):
        self.client = QdrantClient(path=qdrant_path)
        self.sbert = sbert
        self.clip, self.clip_proc = clip_model, clip_proc
        self.slot_c, self.image_c = slot_collection, image_collection
        self.device = device
        self.w = weights or {"garment": 1.0, "scene": 0.6, "vibe": 0.5,
                             "visual": 0.15, "miss_penalty": 0.5}
        self._axis_vec_cache: dict[tuple[str, str], np.ndarray] = {}

    # -- embedding helpers ------------------------------------------------
    def _sbert(self, text: str) -> np.ndarray:
        return self.sbert.encode([text], normalize_embeddings=True)[0]

    def _slot_axis_vec(self, axis: str, value: str) -> np.ndarray:
        """Query value -> unit vector over the axis vocabulary (matches the slot side).

        category is matched semantically (shirt≈blouse≈top); colour and pattern are
        matched exactly (red must not leak into orange). Memoised: queries and the swap
        test hit the same (axis, value) pairs thousands of times.
        """
        cached = self._axis_vec_cache.get((axis, value))
        if cached is None:
            if axis == "category":
                cached = axis_encoding.semantic_vocab_vector(axis, value, self.sbert)
            else:
                cached = axis_encoding.soft_vocab_vector(axis, value, self.sbert)
            self._axis_vec_cache[(axis, value)] = cached
        return cached

    def _clip_text(self, text: str) -> np.ndarray | None:
        if self.clip is None:
            return None
        import torch
        with torch.no_grad():
            t = self.clip_proc(text=[text], return_tensors="pt", padding=True).to(self.device)
            e = self.clip.get_text_features(**t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e[0].cpu().numpy()

    # -- stage 1 ----------------------------------------------------------
    def _candidates(self, q: ParsedQuery, per_axis_k: int) -> set[str]:
        ids: set[str] = set()
        for g in q.garments:
            # Search EVERY specified axis and union: now that colour is a sharp
            # distribution vector, a colour search actually pulls in red garments that a
            # category-only search would miss -- the fix that lets stage-2 rank them.
            for axis, val in g.specified().items():
                vec = self._slot_axis_vec(axis, val)
                hits = self.client.query_points(
                    self.slot_c, query=vec.tolist(), using=axis,
                    limit=per_axis_k, with_payload=["image_id"]).points
                ids.update(h.payload["image_id"] for h in hits)

        for axis in GLOBAL_AXES:
            val = getattr(q, axis)
            if not val:
                continue
            vec = self._sbert(val)
            hits = self.client.query_points(
                self.image_c, query=vec.tolist(), using=axis,
                limit=per_axis_k, with_payload=["image_id"]).points
            ids.update(h.payload["image_id"] for h in hits)
        return ids

    # -- fetch slots + image records for candidates -----------------------
    def _fetch_slots(self, image_ids: list[str]) -> dict[str, list]:
        by_img: dict[str, list] = {i: [] for i in image_ids}
        off = None
        flt = Filter(must=[FieldCondition(key="image_id", match=MatchAny(any=image_ids))])
        while True:
            pts, off = self.client.scroll(
                self.slot_c, scroll_filter=flt, limit=1000, offset=off,
                with_payload=True, with_vectors=True)
            for p in pts:
                by_img.setdefault(p.payload["image_id"], []).append(p)
            if off is None:
                break
        return by_img

    def _fetch_images(self, image_ids: list[str]) -> dict[str, object]:
        flt = Filter(must=[FieldCondition(key="image_id", match=MatchAny(any=image_ids))])
        out = {}
        off = None
        while True:
            pts, off = self.client.scroll(
                self.image_c, scroll_filter=flt, limit=1000, offset=off,
                with_payload=True, with_vectors=True)
            for p in pts:
                out[p.payload["image_id"]] = p
            if off is None:
                break
        return out

    # -- stage 2: bipartite binding --------------------------------------
    def _bind_score(self, q: ParsedQuery, slots: list, qvecs: list[dict]) -> tuple[float, list]:
        """Hungarian assignment between query garments and image slots.

        Cost = 1 - mean cosine over the axes the query specified (zero-weight rule:
        unmentioned axes never enter the mean). Unmatched query garments are penalised
        explicitly, so an image missing the tie cannot tie with one that has it.
        """
        n_q = len(qvecs)
        if n_q == 0:
            return 0.0, []
        if not slots:
            return -self.w["miss_penalty"] * n_q, []

        n_s = len(slots)
        sim = np.zeros((n_q, n_s))
        for i, qv in enumerate(qvecs):
            for j, sp in enumerate(slots):
                cs = []
                for axis, qvec in qv.items():
                    svec = np.array(sp.vector[axis])
                    cs.append(max(0.0, float(qvec @ svec)))   # unit-norm -> cosine, clamp ≥0
                # GEOMETRIC mean, not arithmetic: the axes are an AND. "red tie" must
                # match category AND colour -- a red dress (colour hit, category miss)
                # collapses to ~0 instead of scoring 0.5 for hitting one axis. Kept on a
                # [0,1] scale so single- and multi-axis garments stay comparable.
                sim[i, j] = float(np.prod(cs) ** (1.0 / len(cs))) if cs else 0.0

        cost = 1.0 - sim
        rows, cols = linear_sum_assignment(cost)

        # A requested garment whose best available slot scores below the floor is not
        # actually present in the image -> treat it as a MISS and penalise, instead of
        # averaging in a near-zero sim. Without this, an image that has the white shirt
        # but NOT the red tie keeps a decent garment score and lets the scene term carry
        # it above images that genuinely match the garments.
        floor = self.w.get("match_floor", 0.12)
        matched, total = [], 0.0
        for r, c in zip(rows, cols):
            s = float(sim[r, c])
            if s < floor:
                total -= self.w["miss_penalty"]
                continue
            total += s
            sp = slots[c]
            matched.append({
                "query_garment": q.garments[r].specified(),
                "slot_label": sp.payload["labels"],
                "slot_score": round(s, 3),
                "bbox": sp.payload.get("bbox"),
            })
        # query garments with no slot at all (image has fewer slots than the query asks) are misses too
        total -= self.w["miss_penalty"] * (n_q - len(rows))
        return total / n_q, matched

    # -- public -----------------------------------------------------------
    def search(self, q: ParsedQuery, k: int = 10, per_axis_k: int = 80) -> list[SearchResult]:
        cand_ids = list(self._candidates(q, per_axis_k))
        if not cand_ids:
            return []

        slots_by_img = self._fetch_slots(cand_ids)
        imgs = self._fetch_images(cand_ids)

        # precompute query vectors once (slot axes use vocabulary-distribution space)
        qvecs = [{a: self._slot_axis_vec(a, v) for a, v in g.specified().items()}
                 for g in q.garments]
        scene_vec = self._sbert(q.scene) if q.scene else None
        vibe_vec = self._sbert(q.style_vibe) if q.style_vibe else None
        visual_q = self._clip_text(q.raw) if (self.clip and self.w["visual"] > 0) else None

        results = []
        for iid in cand_ids:
            img = imgs.get(iid)
            if img is None:
                continue
            slots = slots_by_img.get(iid, [])

            g_score, matches = self._bind_score(q, slots, qvecs)
            score = self.w["garment"] * g_score
            bd = {"garment": round(self.w["garment"] * g_score, 3)}

            if scene_vec is not None:
                s = float(scene_vec @ np.array(img.vector["scene"]))
                score += self.w["scene"] * s
                bd["scene"] = round(self.w["scene"] * s, 3)
            if vibe_vec is not None:
                v = float(vibe_vec @ np.array(img.vector["style_vibe"]))
                score += self.w["vibe"] * v
                bd["vibe"] = round(self.w["vibe"] * v, 3)
            if visual_q is not None:
                vg = float(visual_q @ np.array(img.vector["visual_global"]))
                score += self.w["visual"] * vg
                bd["visual"] = round(self.w["visual"] * vg, 3)

            results.append(SearchResult(
                image_id=iid, image_path=img.payload["image_path"], score=round(score, 4),
                source=img.payload["source"], matches=matches,
                scene_label=img.payload["labels"].get("scene"), breakdown=bd))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
