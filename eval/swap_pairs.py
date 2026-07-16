"""Shared construction of the compositional swap pairs (fashion.md sec.6).

Both the G-ADR swap test and the baseline comparison import from here, so every model
is judged on the IDENTICAL set of pairs -- that is what makes the 100%-vs-~50% headline
a fair comparison rather than two tests that happen to disagree.

A pair is an image containing two garments of different category AND different colour.
The TRUE description binds each colour to its own garment; the SWAP exchanges the two
colours. A model that binds must prefer TRUE; a bag-of-words model cannot tell them apart.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

# garment parts / decorations are not bindable garments
NON_GARMENT_LABELS = {"collar", "sleeve", "pocket", "neckline", "zipper", "buckle"}


def load_slots_by_image(client, collection="glance_slots"):
    by_img = defaultdict(list)
    off = None
    while True:
        pts, off = client.scroll(collection, limit=1000, offset=off,
                                 with_payload=True, with_vectors=True)
        for p in pts:
            if p.payload["labels"]["category"] in NON_GARMENT_LABELS:
                continue
            by_img[p.payload["image_id"]].append(p)
        if off is None:
            break
    return by_img


def find_swap_pairs(by_img):
    """-> list of dicts: image_id, image_path, slots, a, b (the two chosen slots)."""
    pairs = []
    for iid, slots in by_img.items():
        for a, b in combinations(slots, 2):
            la, lb = a.payload["labels"], b.payload["labels"]
            if la["category"] != lb["category"] and la["color"] != lb["color"]:
                pairs.append({"image_id": iid,
                              "image_path": a.payload["image_path"],
                              "slots": slots, "a": a, "b": b})
                break
    return pairs


def pair_descriptions(a, b):
    """Natural-language TRUE and SWAP descriptions for a pair (for text-conditioned baselines)."""
    la, lb = a.payload["labels"], b.payload["labels"]
    true = (f"a photo of a person wearing a {la['color']} {la['category']} "
            f"and a {lb['color']} {lb['category']}")
    swap = (f"a photo of a person wearing a {lb['color']} {la['category']} "
            f"and a {la['color']} {lb['category']}")
    return true, swap
