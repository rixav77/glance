"""Phase 0 fallback: build the scene-rich supplement (fashion.md sec.2).

Fashionpedia val is 77% runway/studio -- it has 5 office images and 8 home
images, so eval queries 2/3/4 have nothing to retrieve. This selects real
photos of PEOPLE IN SCENES from COCO to fill the gap.

Cheap by construction: COCO train2017 images are 18GB, so instead of pulling
them all we (1) text-filter captions for person + scene words, (2) download
only those candidates individually, (3) CLIP-verify the scene and keep the
confident ones.

Stage 1 (this script, no GPU): caption filter -> candidate URL list.
Stage 2 (fetch_coco_images.py): download + CLIP scene verification.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco" / "annotations"
OUT = ROOT / "data" / "coco_candidates.json"

# Scene words as they actually appear in COCO captions. Deliberately generous
# here -- CLIP verification in stage 2 is what enforces precision.
SCENE_WORDS = {
    "office": [
        r"\boffice\b", r"\bdesk\b", r"\bcubicle\b", r"\bconference room\b",
        r"\bmeeting room\b", r"\bboardroom\b", r"\bworkplace\b", r"\bbusiness meeting\b",
    ],
    "home": [
        r"\bliving room\b", r"\bkitchen\b", r"\bbedroom\b", r"\bcouch\b", r"\bsofa\b",
        r"\bat home\b", r"\bdining room\b", r"\bhome\b",
    ],
    "urban street": [
        r"\bstreet\b", r"\bsidewalk\b", r"\bcrosswalk\b", r"\bcity\b",
        r"\bdowntown\b", r"\bintersection\b", r"\burban\b",
    ],
    "park": [
        r"\bpark\b", r"\bpark bench\b", r"\bgarden\b", r"\bgrass\b",
        r"\bfield\b", r"\bpicnic\b", r"\bmeadow\b",
    ],
}
PERSON_WORDS = re.compile(
    r"\b(man|woman|person|people|guy|lady|boy|girl|men|women|someone|he|she)\b", re.I
)
# How many candidates to keep per scene before CLIP verification. Overshoot,
# because verification will reject a good fraction.
PER_SCENE = 400


def main() -> None:
    caps = json.load((COCO / "captions_train2017.json").open())
    imgs = {im["id"]: im for im in caps["images"]}

    # image_id -> all its captions joined
    text = defaultdict(list)
    for a in caps["annotations"]:
        text[a["image_id"]].append(a["caption"])

    # COCO instances tell us which images genuinely contain a person, which is
    # far more reliable than trusting the caption to mention one.
    inst = json.load((COCO / "instances_train2017.json").open())
    person_cat = next(c["id"] for c in inst["categories"] if c["name"] == "person")
    # Require a reasonably large person: tiny background figures show no clothing.
    big_person = {
        a["image_id"] for a in inst["annotations"]
        if a["category_id"] == person_cat and a.get("area", 0) > 20000
    }
    print(f"COCO train2017 images:            {len(imgs)}")
    print(f"images with a large (>20k px) person: {len(big_person)}")

    patterns = {s: [re.compile(p, re.I) for p in ps] for s, ps in SCENE_WORDS.items()}
    hits = defaultdict(list)
    for img_id in big_person:
        blob = " ".join(text.get(img_id, []))
        if not PERSON_WORDS.search(blob):
            continue
        for scene, pats in patterns.items():
            if any(p.search(blob) for p in pats):
                hits[scene].append(img_id)
                break  # first scene wins; CLIP will arbitrate properly in stage 2

    print("\ncaption-filtered candidates (pre-CLIP):")
    cands = []
    for scene in SCENE_WORDS:
        ids = hits[scene][:PER_SCENE]
        print(f"  {scene:<14} {len(hits[scene]):>5} found -> keeping {len(ids)}")
        for i in ids:
            cands.append({
                "image_id": i,
                "file_name": f"coco_{imgs[i]['file_name']}",
                "url": imgs[i]["coco_url"],
                "caption_scene": scene,
            })

    OUT.write_text(json.dumps(cands, indent=2))
    print(f"\ntotal candidates: {len(cands)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
