"""Phase 0 audit, part 2: is the `scene` axis usable on Fashionpedia val?

The go/no-go gate from fashion.md sec.2. Eval queries 2, 3 and 4 need offices,
parks and city streets. Fashionpedia is fashion photography, so the corpus may
be dominated by studio backdrops and runways -- in which case those three
queries have nothing to retrieve and we must supplement the dataset.

Runs CLIP zero-shot scene classification over all 1158 val images and reports
the scene histogram, its normalised entropy, and the studio+runway share.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "instances_attributes_val2020.json"
IMG_DIR = ROOT / "data" / "test"
OUT = ROOT / "eval" / "scene_audit.json"

# The four scenes the assignment demands, plus the ones we suspect dominate.
SCENES = {
    "office": "a photo taken inside an office",
    "urban street": "a photo taken on an urban city street",
    "park": "a photo taken in a park or garden",
    "home": "a photo taken inside a home",
    "studio": "a studio photo on a plain seamless backdrop",
    "runway": "a photo of a fashion runway show",
    "beach": "a photo taken at the beach",
    "restaurant/cafe": "a photo taken inside a restaurant or cafe",
}
MODEL = "openai/clip-vit-base-patch32"
BATCH = 64


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    names, prompts = list(SCENES), list(SCENES.values())
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
        temb = model.get_text_features(**t)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    files = [im["file_name"] for im in json.load(ANN.open())["images"]]
    print(f"images: {len(files)}")

    rows, top = [], Counter()
    for i in range(0, len(files), BATCH):
        chunk = files[i : i + BATCH]
        imgs = [Image.open(IMG_DIR / f).convert("RGB") for f in chunk]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt").to(dev)
            iemb = model.get_image_features(**px)
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            probs = (100.0 * iemb @ temb.T).softmax(dim=-1).cpu()

        for f, p in zip(chunk, probs):
            k = int(p.argmax())
            top[names[k]] += 1
            rows.append({"file": f, "scene": names[k], "conf": round(float(p[k]), 4)})
        print(f"  {min(i+BATCH, len(files))}/{len(files)}", flush=True)

    n = len(rows)
    print("\n=== SCENE HISTOGRAM ===")
    for s, c in top.most_common():
        print(f"  {s:<18} {c:>5}  ({100*c/n:5.1f}%)")

    # Normalised entropy: 1.0 = perfectly uniform, 0.0 = one scene only.
    import math
    ent = -sum((c / n) * math.log(c / n) for c in top.values() if c)
    ent /= math.log(len(SCENES))
    dominated = 100 * (top["studio"] + top["runway"]) / n
    required = {s: 100 * top[s] / n for s in ("office", "urban street", "park", "home")}

    print(f"\nnormalised entropy:   {ent:.3f}   (1.0 = uniform, 0.0 = collapsed)")
    print(f"studio + runway:      {dominated:.1f}%")
    print("required scenes (assignment queries 2/3/4):")
    for s, v in required.items():
        print(f"  {s:<18} {v:5.1f}%  {'OK' if v >= 15 else 'THIN'}")

    verdict = (
        "PROCEED on Fashionpedia alone"
        if all(v >= 15 for v in required.values())
        else "SUPPLEMENT with scene-rich images (fashion.md sec.2 fallback)"
    )
    print(f"\n>>> VERDICT: {verdict}")

    OUT.write_text(json.dumps(
        {"histogram": dict(top), "entropy": ent, "studio_runway_pct": dominated,
         "required": required, "verdict": verdict, "per_image": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
