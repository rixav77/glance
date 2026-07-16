"""The compositional swap test (fashion.md sec.6) -- the metric that proves binding.

Corpus-wide retrieval is a poor binding probe: if no image happens to contain "red
tie + white shirt", the ideal result simply is not there to rank, and half-matches tie
with half-matches. So we test binding the honest way -- PER IMAGE:

  Take images that genuinely contain two garments of DIFFERENT colours. For each, build
  the TRUE query (each colour bound to its own garment) and the SWAPPED query (the two
  colours exchanged). A model that binds colour to garment must score TRUE > SWAP on the
  SAME image. A bag-of-words model scores them equal, because both contain the same words.

Reports swap accuracy = fraction of pairs with TRUE > SWAP. Chance is 50%.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

from eval.swap_pairs import find_swap_pairs, load_slots_by_image
from retriever.search import Retriever
from shared.schema import GarmentConstraint, ParsedQuery

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = yaml.safe_load((ROOT / "retriever" / "config.yaml").read_text())
    dev = "cpu"
    sbert = SentenceTransformer(cfg["models"]["sbert"], device=dev)
    retr = Retriever(str(ROOT / "data" / "qdrant"), sbert,
                     weights=cfg["weights"], device=dev)   # no CLIP: garment term only

    client = retr.client   # reuse: qdrant local mode allows only one client per path

    # identical pair construction to the baselines, via the shared module
    raw_pairs = find_swap_pairs(load_slots_by_image(client))
    pairs = [(p["image_id"], p["slots"], p["a"], p["b"]) for p in raw_pairs]
    print(f"images with a usable swap pair: {len(pairs)}")

    def score(slots, garments):
        q = ParsedQuery(raw="", garments=garments)
        qv = [{ax: retr._slot_axis_vec(ax, v) for ax, v in g.specified().items()}
              for g in garments]
        s, _ = retr._bind_score(q, slots, qv)
        return s

    wins = ties = 0
    margins = []
    for iid, slots, a, b in pairs:
        la, lb = a.payload["labels"], b.payload["labels"]
        true_q = [GarmentConstraint(category=la["category"], color=la["color"]),
                  GarmentConstraint(category=lb["category"], color=lb["color"])]
        swap_q = [GarmentConstraint(category=la["category"], color=lb["color"]),
                  GarmentConstraint(category=lb["category"], color=la["color"])]
        st, ss = score(slots, true_q), score(slots, swap_q)
        margins.append(st - ss)
        if st > ss + 1e-6:
            wins += 1
        elif abs(st - ss) <= 1e-6:
            ties += 1

    n = len(pairs)
    print(f"\nSWAP ACCURACY: {wins}/{n} = {100*wins/n:.1f}%   (chance = 50%)")
    print(f"  ties (identical score -- the bag-of-words failure): {ties} ({100*ties/n:.1f}%)")
    print(f"  mean margin (true - swap): {np.mean(margins):+.4f}")
    print(f"  median margin:             {np.median(margins):+.4f}")

    # a couple of worked examples
    print("\nexamples:")
    for iid, slots, a, b in pairs[:5]:
        la, lb = a.payload["labels"], b.payload["labels"]
        true_q = [GarmentConstraint(la["category"], la["color"]),
                  GarmentConstraint(lb["category"], lb["color"])]
        swap_q = [GarmentConstraint(la["category"], lb["color"]),
                  GarmentConstraint(lb["category"], la["color"])]
        st, ss = score(slots, true_q), score(slots, swap_q)
        mark = "OK " if st > ss else "XX "
        print(f"  {mark}{la['color']} {la['category']} + {lb['color']} {lb['category']}"
              f"   true={st:.3f}  swap={ss:.3f}")


if __name__ == "__main__":
    main()
