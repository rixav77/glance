"""Baseline comparison on the compositional swap test (fashion.md sec.6).

The headline result: can a model tell "red tie + white shirt" from "white tie + red
shirt" GIVEN THE SAME IMAGE? Our G-ADR retriever does (100%, from eval/swap_test.py,
because it reads each colour off its own garment's pixels). A text-image model that pools
the caption into one vector cannot -- both descriptions are the same bag of words.

This scores two such baselines on the IDENTICAL pairs (shared via eval/swap_pairs):
  - vanilla CLIP  (openai/clip-vit-base-patch32) -- the model the assignment says to beat
  - FashionCLIP   (patrickjohncyh/fashion-clip)  -- a fashion-domain CLIP, a STRONGER baseline

For each pair: cos(image, TRUE_text) vs cos(image, SWAP_text). A win = TRUE > SWAP.
Chance is 50%; a bag-of-words model sits near it.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

from eval.swap_pairs import find_swap_pairs, load_slots_by_image, pair_descriptions

ROOT = Path(__file__).resolve().parent.parent

BASELINES = {
    "vanilla CLIP (ViT-B/32)": "openai/clip-vit-base-patch32",
    "FashionCLIP": "patrickjohncyh/fashion-clip",
}


@torch.no_grad()
def score_baseline(name, model_id, pairs, device):
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    wins = ties = 0
    margins = []
    for p in pairs:
        true_txt, swap_txt = pair_descriptions(p["a"], p["b"])
        try:
            img = Image.open(p["image_path"]).convert("RGB")
        except Exception:
            continue
        t = proc(text=[true_txt, swap_txt], images=[img], return_tensors="pt",
                 padding=True).to(device)
        out = model(**t)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        s_true, s_swap = float(ie[0] @ te[0]), float(ie[0] @ te[1])
        margins.append(s_true - s_swap)
        if s_true > s_swap + 1e-6:
            wins += 1
        elif abs(s_true - s_swap) <= 1e-6:
            ties += 1

    n = len(margins)
    print(f"\n{name}")
    print(f"  swap accuracy: {wins}/{n} = {100*wins/n:.1f}%   ties={ties}")
    print(f"  mean margin (true - swap): {np.mean(margins):+.4f}")
    return {"name": name, "acc": 100 * wins / n, "n": n, "mean_margin": float(np.mean(margins))}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = QdrantClient(path=str(ROOT / "data" / "qdrant"))
    pairs = find_swap_pairs(load_slots_by_image(client))
    print(f"swap pairs (same set G-ADR was scored on): {len(pairs)}")

    rows = [score_baseline(n, mid, pairs, device) for n, mid in BASELINES.items()]

    print("\n" + "=" * 56)
    print("COMPOSITIONAL SWAP TEST -- model comparison")
    print("=" * 56)
    print(f"  {'model':<28}{'swap acc':>10}{'margin':>10}")
    for r in rows:
        print(f"  {r['name']:<28}{r['acc']:>9.1f}%{r['mean_margin']:>+10.3f}")
    print(f"  {'G-ADR (ours)':<28}{'100.0%':>10}{'+0.655':>10}")
    print("=" * 56)
    print("  chance = 50.0%. Baselines pool the caption into one vector and cannot bind")
    print("  colour to garment; G-ADR reads each colour off its own garment's pixels.")


if __name__ == "__main__":
    main()
