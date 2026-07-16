"""Search smoke test: does structural binding actually discriminate the swap?

Runs the compositional query and its colour-swapped twin through the SAME retriever
(no LLM -- constructs the ParsedQuery directly so we test search, not parsing). If
binding works, "red tie + white shirt" should rank tie-is-red/shirt-is-white images
above the swap, and vice versa.
"""
import sys, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from retriever.search import Retriever
from shared.schema import GarmentConstraint, ParsedQuery

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "retriever" / "config.yaml").read_text())
dev = "cuda" if torch.cuda.is_available() else "cpu"

sbert = SentenceTransformer(cfg["models"]["sbert"], device=dev)
clip = CLIPModel.from_pretrained(cfg["models"]["clip"]).to(dev).eval()
clip_proc = CLIPProcessor.from_pretrained(cfg["models"]["clip"])
retr = Retriever(str(ROOT / "data" / "qdrant"), sbert, clip, clip_proc,
                 weights=cfg["weights"], device=dev)

def run(label, garments, scene=None, vibe=None):
    q = ParsedQuery(raw=label, garments=garments, scene=scene, style_vibe=vibe)
    res = retr.search(q, k=5, per_axis_k=cfg["search"]["per_axis_k"])
    print(f"\n=== {label} ===")
    for i, r in enumerate(res, 1):
        binds = "; ".join(f"{m['slot_label'].get('category')}:{m['slot_label'].get('color')}"
                          f"~{m['slot_score']}" for m in r.matches)
        print(f" #{i} {r.score:<6} [{r.source}] scene={r.scene_label} :: {binds}")
    return res

# Query 1 -- sanity
run("bright yellow raincoat", [GarmentConstraint(category="raincoat", color="yellow")])
# Query 5 and its swap -- the binding test
run("red tie AND white shirt", [GarmentConstraint(category="tie", color="red"),
                                 GarmentConstraint(category="shirt", color="white")], vibe="formal")
run("white tie AND red shirt (SWAP)", [GarmentConstraint(category="tie", color="white"),
                                       GarmentConstraint(category="shirt", color="red")], vibe="formal")
# Query 2 -- scene/vibe only, zero-weight rule
run("office + professional (no garment)", [], scene="office", vibe="business professional")
print("\nOK")
