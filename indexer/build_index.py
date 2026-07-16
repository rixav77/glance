"""Part A: the indexer. Corpus -> garment slots -> Qdrant.

Writes two collections (fashion.md sec.3.6):

  glance_slots  -- ONE POINT PER GARMENT REGION, with named vectors for category,
                   colour and pattern. One point per *slot* rather than per *image*
                   is what lets a variable number of garments live in a fixed-schema
                   vector DB with no custom fusion code, and it makes stage-1 candidate
                   generation a single ANN call.

  glance_images -- one point per image: scene, style_vibe, visual_global.

Usage:
    python -m indexer.build_index                       # ships: detector regions
    python -m indexer.build_index --region fashionpedia_gt --collection-suffix _gt
    python -m indexer.build_index --limit 50            # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import torch
import yaml
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer.attributes import AttributeExtractor
from indexer.corpus import load_corpus
from indexer.regions import build_region_source
from shared import axis_encoding
from shared.schema import CLIP_DIM, GLOBAL_AXES, SBERT_DIM, SLOT_AXES

ROOT = Path(__file__).resolve().parent.parent


def load_models(cfg: dict, device: str):
    from sentence_transformers import SentenceTransformer
    from transformers import (BlipForQuestionAnswering, BlipProcessor,
                              CLIPModel, CLIPProcessor)

    m = cfg["models"]
    clip = CLIPModel.from_pretrained(m["clip"]).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(m["clip"])
    sbert = SentenceTransformer(m["sbert"], device=device)

    vqa = vqa_proc = None
    if m.get("vqa"):
        vqa_proc = BlipProcessor.from_pretrained(m["vqa"])
        vqa = BlipForQuestionAnswering.from_pretrained(m["vqa"]).to(device).eval()
    return clip, clip_proc, sbert, vqa, vqa_proc


def ensure_collections(client: QdrantClient, slot_c: str, image_c: str) -> None:
    for name in (slot_c, image_c):
        if client.collection_exists(name):
            client.delete_collection(name)

    # Slot axes are vocabulary distributions -> one dim per vocabulary label
    # (category/colour/pattern), NOT the SBERT dim. See shared/axis_encoding.
    slot_dims = axis_encoding.slot_axis_dims()
    client.create_collection(
        slot_c,
        vectors_config={a: VectorParams(size=slot_dims[a], distance=Distance.COSINE)
                        for a in SLOT_AXES},
    )
    client.create_collection(
        image_c,
        vectors_config={
            **{a: VectorParams(size=SBERT_DIM, distance=Distance.COSINE)
               for a in GLOBAL_AXES},
            "visual_global": VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "indexer" / "config.yaml"))
    ap.add_argument("--region", default=None, help="override region_source.kind")
    ap.add_argument("--collection-suffix", default="", help="e.g. _gt for the oracle index")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on N images")
    ap.add_argument("--no-vqa", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.region:
        cfg["region_source"]["kind"] = args.region
    if args.no_vqa:
        cfg["models"]["vqa"] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"region source: {cfg['region_source']['kind']}")

    corpus = load_corpus(tuple(cfg["corpus"]["include"]))
    if args.limit:
        corpus = corpus[: args.limit]
    print(f"corpus: {len(corpus)} images")

    clip, clip_proc, sbert, vqa, vqa_proc = load_models(cfg, device)
    ex = AttributeExtractor(
        clip, clip_proc, sbert, device=device,
        color_lambda=cfg["extraction"]["color_lambda"],
        entropy_threshold=cfg["extraction"]["entropy_threshold"],
        vqa=vqa, vqa_proc=vqa_proc,
        temperature=cfg["extraction"]["clip_temperature"],
        fallback_axes=tuple(cfg["extraction"].get("fallback_axes", ["color"])),
    )

    kind = cfg["region_source"]["kind"]
    regions = build_region_source(
        kind,
        ann_path=ROOT / "data" / "instances_attributes_val2020.json",
        device=device,
        min_area_frac=cfg["region_source"]["min_area_frac"],
    )

    qcfg = cfg["qdrant"]
    slot_c = qcfg["slot_collection"] + args.collection_suffix
    image_c = qcfg["image_collection"] + args.collection_suffix
    client = QdrantClient(path=str(ROOT / "data" / "qdrant"))
    ensure_collections(client, slot_c, image_c)

    slot_pts: list[PointStruct] = []
    img_pts: list[PointStruct] = []
    n_slots = 0
    no_slot_imgs = 0
    fallback_hits = 0
    t0 = time.time()

    for i, ci in enumerate(corpus):
        try:
            img = Image.open(ci.path).convert("RGB")
        except Exception as e:
            print(f"  !! skip {ci.image_id}: {e}")
            continue

        # -- global axes ---------------------------------------------------
        g = ex.global_attributes(img)

        # -- garment slots (the binding unit) ------------------------------
        # The GT source keys off the Fashionpedia file name; the detector keys off
        # nothing at all, which is precisely why it generalises to the supplement.
        key = ci.fp_file_name if ci.source == "fashionpedia" else ci.image_id
        regs = regions.regions(key, img)
        if not regs:
            no_slot_imgs += 1

        slot_ids: list[str] = []
        for r in regs:
            a = ex.slot_attributes(r, img)
            sid = str(uuid.uuid4())
            slot_ids.append(sid)
            fallback_hits += len(a["fallback_axes"])
            slot_pts.append(PointStruct(
                id=sid,
                vector={ax: a["embeddings"][ax].tolist() for ax in SLOT_AXES},
                payload={
                    "image_id": ci.image_id,
                    "image_path": str(ci.path),
                    "source": ci.source,
                    "bbox": list(r.bbox),
                    "area_frac": round(r.area_frac, 4),
                    "labels": a["labels"],
                    "fallback_axes": a["fallback_axes"],
                    "gt_category": r.gt_category,
                    "region_label": r.source_label,
                },
            ))
            n_slots += 1

        img_pts.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, ci.image_id)),
            vector={
                **{ax: g["embeddings"][ax].tolist() for ax in GLOBAL_AXES},
                "visual_global": g["visual_global"].tolist(),
            },
            payload={
                "image_id": ci.image_id,
                "image_path": str(ci.path),
                "source": ci.source,
                "slot_ids": slot_ids,
                "n_slots": len(slot_ids),
                "labels": g["labels"],
                "known_scene": ci.known_scene,
            },
        ))

        bs = qcfg["batch_size"]
        if len(slot_pts) >= bs:
            client.upsert(slot_c, points=slot_pts); slot_pts = []
        if len(img_pts) >= bs:
            client.upsert(image_c, points=img_pts); img_pts = []

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(corpus)}  slots={n_slots}  "
                  f"{el/(i+1):.2f}s/img  eta={(len(corpus)-i-1)*el/(i+1)/60:.1f}min",
                  flush=True)

    if slot_pts:
        client.upsert(slot_c, points=slot_pts)
    if img_pts:
        client.upsert(image_c, points=img_pts)

    print(f"\nindexed {len(corpus)} images -> {n_slots} slots "
          f"({n_slots/max(len(corpus),1):.1f} slots/image)")
    print(f"images with ZERO slots: {no_slot_imgs} "
          f"({100*no_slot_imgs/max(len(corpus),1):.1f}%)  <- these are unreachable by "
          f"any garment query")
    print(f"VQA fallback fired on {fallback_hits} slot-axes")
    print(f"collections: {slot_c}, {image_c}")
    print(f"took {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
