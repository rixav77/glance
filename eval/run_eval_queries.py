"""Run the five assignment queries through the full pipeline, dump results to JSON.

Feeds the visual contact sheet. Captures, per result, the image path, score breakdown,
and the per-garment BINDING detail (which slot each query garment matched) -- so the
contact sheet can show WHY each image was retrieved, not just that it was.
"""

import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever.query_parser import QueryParser
from retriever.search import Retriever

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "eval_queries_results.json"

QUERIES = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]
K = 5


def main() -> None:
    cfg = yaml.safe_load((ROOT / "retriever" / "config.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

    sbert = SentenceTransformer(cfg["models"]["sbert"], device=device)
    clip = CLIPModel.from_pretrained(cfg["models"]["clip"]).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(cfg["models"]["clip"])
    parser = QueryParser(cfg["models"]["parser_llm"], device=device, use_llm=True)
    retr = Retriever(str(ROOT / "data" / "qdrant"), sbert, clip, clip_proc,
                     weights=cfg["weights"], device=device)

    out = []
    for q in QUERIES:
        pq = parser.parse(q)
        results = retr.search(pq, k=K, per_axis_k=cfg["search"]["per_axis_k"])
        out.append({
            "query": q,
            "parsed": {
                "garments": [g.specified() for g in pq.garments],
                "scene": pq.scene, "style_vibe": pq.style_vibe,
            },
            "results": [{
                "rank": i + 1,
                "image_path": r.image_path,
                "source": r.source,
                "score": r.score,
                "scene_label": r.scene_label,
                "breakdown": {k: float(v) for k, v in r.breakdown.items()},
                "matches": [{
                    "query_garment": m["query_garment"],
                    "slot_label": {k: m["slot_label"].get(k)
                                   for k in ("category", "color", "pattern")},
                    "slot_score": m["slot_score"],
                    "bbox": m["bbox"],
                } for m in r.matches],
            } for i, r in enumerate(results)],
        })
        print(f"done: {q}  -> {len(results)} results", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
