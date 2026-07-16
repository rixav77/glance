"""Retrieval CLI.

    python -m retriever.cli "a red tie and a white shirt in a formal setting" --k 10
    python -m retriever.cli "casual weekend outfit for a city walk" --no-llm

Prints the parsed structured query (so you can see the decomposition), then the ranked
results with a per-garment binding breakdown -- which slot each query garment matched
and how well, making the structural binding visible rather than a black box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever.query_parser import QueryParser
from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent.parent


def load(cfg, use_llm, device):
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    m = cfg["models"]
    sbert = SentenceTransformer(m["sbert"], device=device)
    clip = CLIPModel.from_pretrained(m["clip"]).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(m["clip"])
    parser = QueryParser(m["parser_llm"], device=device, use_llm=use_llm)
    retr = Retriever(
        qdrant_path=str(ROOT / "data" / "qdrant"), sbert=sbert,
        clip_model=clip, clip_proc=clip_proc,
        slot_collection=cfg["qdrant"]["slot_collection"],
        image_collection=cfg["qdrant"]["image_collection"],
        weights=cfg["weights"], device=device)
    return parser, retr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "retriever" / "config.yaml"))
    ap.add_argument("--no-llm", action="store_true", help="use the vocab-scan parser")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    k = args.k or cfg["search"]["default_k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser, retr = load(cfg, use_llm=not args.no_llm, device=device)

    q = parser.parse(args.query)
    print(f"\nQUERY: {q.raw}")
    print("PARSED:")
    for i, g in enumerate(q.garments):
        print(f"  garment[{i}]: {g.specified()}")
    if q.scene:      print(f"  scene: {q.scene}")
    if q.style_vibe: print(f"  style_vibe: {q.style_vibe}")
    if q.is_empty(): print("  (empty parse)")

    results = retr.search(q, k=k, per_axis_k=cfg["search"]["per_axis_k"])
    print(f"\nTOP {len(results)} of {k} requested:")
    for rank, r in enumerate(results, 1):
        fname = Path(r.image_path).name
        print(f"\n#{rank}  score={r.score:<7} [{r.source}] scene={r.scene_label}  {fname}")
        print(f"     breakdown: {r.breakdown}")
        for mt in r.matches:
            g = mt["query_garment"]
            sl = {k2: mt["slot_label"].get(k2) for k2 in ("category", "color", "pattern")}
            print(f"     bind {g}  ->  slot {sl}  (sim={mt['slot_score']})")


if __name__ == "__main__":
    main()
