"""The corpus: Fashionpedia core + CLIP-verified COCO scene supplement.

Composition is deliberate and documented (fashion.md sec.2). Fashionpedia carries the
garments (4.7 per image, real masks, real pattern GT) but is 77% runway/studio and has
FIVE office images -- so on its own, eval queries 2/3/4 have nothing to retrieve. The
COCO supplement carries the scenes (322 office / 301 street / 249 park / 245 home).

Neither half is sufficient. The union is the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class CorpusImage:
    image_id: str          # unique across both sources
    path: Path
    source: str            # "fashionpedia" | "coco_supplement"
    fp_file_name: str | None = None    # key into the Fashionpedia GT annotations
    known_scene: str | None = None     # CLIP-verified scene (supplement only)


def load_corpus(include: tuple[str, ...] = ("fashionpedia", "coco_supplement")) -> list[CorpusImage]:
    out: list[CorpusImage] = []

    if "fashionpedia" in include:
        ann = json.loads((DATA / "instances_attributes_val2020.json").read_text())
        for im in ann["images"]:
            p = DATA / "test" / im["file_name"]
            if p.exists():
                out.append(CorpusImage(
                    image_id=f"fp_{im['id']}", path=p, source="fashionpedia",
                    fp_file_name=im["file_name"],
                ))

    if "coco_supplement" in include:
        man = json.loads((DATA / "supplement_manifest.json").read_text())
        for c in man:
            p = DATA / "supplement" / c["file_name"]
            if p.exists():
                out.append(CorpusImage(
                    image_id=f"coco_{c['image_id']}", path=p, source="coco_supplement",
                    known_scene=c["scene"],
                ))

    return out


if __name__ == "__main__":
    from collections import Counter

    c = load_corpus()
    print(f"corpus: {len(c)} images")
    for s, n in Counter(i.source for i in c).most_common():
        print(f"  {s:<18} {n}")
