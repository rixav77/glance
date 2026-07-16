"""Phase 0 fallback, stage 2: download COCO candidates and CLIP-verify their scenes.

The caption filter in build_coco_supplement.py is generous on purpose (a caption
saying "home" may show a porch, not a living room). This is the precision gate:
CLIP classifies each downloaded image against the SAME scene prompt bank the
indexer will use, and we keep only images whose CLIP scene matches the scene the
caption promised, above a confidence floor.

That agreement requirement is the point -- it means every supplement image is one
that BOTH a human captioner and CLIP call e.g. "office", so the scene axis has a
trustworthy population to retrieve from.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
CANDS = ROOT / "data" / "coco_candidates.json"
IMG_DIR = ROOT / "data" / "supplement"
OUT = ROOT / "data" / "supplement_manifest.json"

# Identical to the prompt bank in audit_scenes.py -- indexer and audit must agree.
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
CONF_FLOOR = 0.30   # CLIP must be at least this sure of the scene it picks
BATCH = 64


def download(c: dict) -> dict | None:
    dst = IMG_DIR / c["file_name"]
    if dst.exists():
        return c
    try:
        r = requests.get(c["url"], timeout=20)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return c
    except Exception:
        return None


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    cands = json.loads(CANDS.read_text())
    print(f"candidates: {len(cands)}")

    with ThreadPoolExecutor(max_workers=16) as ex:
        got = [c for c in ex.map(download, cands) if c]
    print(f"downloaded: {len(got)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    names, prompts = list(SCENES), list(SCENES.values())
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
        temb = model.get_text_features(**t)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    kept, rejected = [], Counter()
    for i in range(0, len(got), BATCH):
        chunk = got[i : i + BATCH]
        imgs = [Image.open(IMG_DIR / c["file_name"]).convert("RGB") for c in chunk]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt").to(dev)
            iemb = model.get_image_features(**px)
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            probs = (100.0 * iemb @ temb.T).softmax(dim=-1).cpu()

        for c, p in zip(chunk, probs):
            k = int(p.argmax())
            conf = float(p[k])
            if names[k] == c["caption_scene"] and conf >= CONF_FLOOR:
                kept.append({**c, "scene": names[k], "scene_conf": round(conf, 4)})
            else:
                rejected[f"{c['caption_scene']} -> {names[k]}"] += 1
        print(f"  {min(i+BATCH, len(got))}/{len(got)}", flush=True)

    print(f"\nkept (caption and CLIP agree, conf>={CONF_FLOOR}): {len(kept)}")
    hist = Counter(c["scene"] for c in kept)
    for s, n in hist.most_common():
        print(f"  {s:<14} {n}")

    print("\ntop rejections (caption said X, CLIP said Y):")
    for k, n in rejected.most_common(8):
        print(f"  {k:<34} {n}")

    # Drop the images we rejected, so data/supplement/ IS the corpus.
    keep_names = {c["file_name"] for c in kept}
    removed = 0
    for p in IMG_DIR.glob("coco_*.jpg"):
        if p.name not in keep_names:
            p.unlink()
            removed += 1
    print(f"\nremoved {removed} rejected images from disk")

    OUT.write_text(json.dumps(kept, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
