"""Phase 0 audit, part 1: is the compositional swap test powered?

Answers, from annotations alone (no images needed):
  - do the val annotations resolve against the downloaded image files
  - how many images carry >=2 garment instances with DIFFERENT colours
    (this is the population the swap test of fashion.md sec.6 draws from)
"""

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "instances_attributes_val2020.json"
IMG_DIR = ROOT / "data" / "test"

COLOR_HINTS = ("colour", "color")


def main() -> None:
    with ANN.open() as f:
        d = json.load(f)

    images = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    attrs = {a["id"]: a for a in d["attributes"]}

    print(f"images:      {len(images)}")
    print(f"categories:  {len(cats)}")
    print(f"attributes:  {len(attrs)}")
    print(f"annotations: {len(d['annotations'])}")

    # --- do annotations resolve to files on disk? -------------------------
    on_disk = {p.name for p in IMG_DIR.glob("*.jpg")}
    resolved = sum(1 for im in images.values() if im["file_name"] in on_disk)
    print(f"\nfiles on disk: {len(on_disk)}")
    print(f"val annotations resolving to a file: {resolved}/{len(images)}")

    # --- locate colour attributes ----------------------------------------
    supercats = Counter(a.get("supercategory", "?") for a in attrs.values())
    print("\nattribute supercategories:")
    for sc, n in supercats.most_common():
        print(f"  {sc:<34} {n}")

    color_ids = {
        aid for aid, a in attrs.items()
        if any(h in a.get("supercategory", "").lower() for h in COLOR_HINTS)
        or any(h in a.get("name", "").lower() for h in COLOR_HINTS)
    }
    print(f"\ncolour attribute ids: {len(color_ids)}")
    if color_ids:
        print("  " + ", ".join(sorted(attrs[i]["name"] for i in color_ids)))

    # --- per-image garment slots -----------------------------------------
    per_image = defaultdict(list)
    for ann in d["annotations"]:
        colors = [attrs[a]["name"] for a in ann.get("attribute_ids", []) if a in color_ids]
        per_image[ann["image_id"]].append(
            {"cat": cats[ann["category_id"]], "colors": colors}
        )

    n_colored = sum(1 for v in per_image.values() if any(s["colors"] for s in v))
    print(f"\nimages with >=1 colour-attributed garment: {n_colored}")

    # --- THE number: swap-test population ---------------------------------
    # Need >=2 garments of DIFFERENT categories with DISJOINT colour sets, so
    # that swapping the two colours yields a genuinely wrong query.
    swap_ready = []
    for img_id, slots in per_image.items():
        colored = [s for s in slots if s["colors"]]
        for a, b in combinations(colored, 2):
            if a["cat"] != b["cat"] and set(a["colors"]).isdisjoint(b["colors"]):
                swap_ready.append((img_id, a, b))
                break

    pct = 100 * len(swap_ready) / max(len(images), 1)
    print(f"\n>>> SWAP-TEST POPULATION: {len(swap_ready)} images ({pct:.1f}% of val)")
    print("    (>=2 garments, different categories, disjoint colours)")
    for img_id, a, b in swap_ready[:8]:
        print(f"      {images[img_id]['file_name']:<18} "
              f"{a['colors'][0]} {a['cat']}  +  {b['colors'][0]} {b['cat']}")

    # --- inventory --------------------------------------------------------
    cat_counts = Counter(s["cat"] for v in per_image.values() for s in v)
    print("\ntop garment categories:")
    for c, n in cat_counts.most_common(12):
        print(f"  {c:<28} {n}")

    mean_slots = sum(len(v) for v in per_image.values()) / max(len(per_image), 1)
    print(f"\nmean annotated garments per image: {mean_slots:.1f}")


if __name__ == "__main__":
    main()
