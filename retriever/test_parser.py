"""Exercise the LLM query parser on the five assignment queries end-to-end."""
import sys, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retriever.query_parser import QueryParser

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "retriever" / "config.yaml").read_text())

QUERIES = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
]

p = QueryParser(cfg["models"]["parser_llm"],
                device="cuda" if torch.cuda.is_available() else "cpu", use_llm=True)
for q in QUERIES:
    pq = p.parse(q)
    print(f"\nQ: {q}")
    for i, g in enumerate(pq.garments):
        print(f"   garment[{i}]: {g.specified()}")
    print(f"   scene={pq.scene}  vibe={pq.style_vibe}")
print("\nparser OK")
