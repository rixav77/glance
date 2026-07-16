# Glance — Grounded Axis-Decomposed Retrieval for Fashion & Context

Retrieve fashion images from natural-language queries that mix **garment attributes**,
**compositional binding**, **scene/context**, and **style/vibe** — e.g.

> *"a red tie and a white shirt in a formal setting"* · *"someone in a blue shirt on a park bench"* · *"casual weekend outfit for a city walk"*

The hard part is **compositional binding**: telling *"red tie + white shirt"* apart from
*"white tie + red shirt"*. A pooled text-image model (vanilla CLIP) cannot — both are the
same bag of words. Glance can, because it reads each attribute off *its own garment's
pixels*. On a 1,259-image compositional swap test, Glance scores **100%** where vanilla
CLIP and FashionCLIP sit near chance (see `eval/`).

---

## Design evolution (my own iterations)

This design is the result of iterating on my own earlier attempts — the full ladder, tradeoffs, and
why each was rejected are in [`docs/APPROACHES.md`](docs/APPROACHES.md):

1. **Naive tries** — vanilla CLIP, a fashion-tuned CLIP, and caption-and-match. All pool into one
   vector → bag-of-words → fail the compositional case.
2. **v1 — ADR-global (my first real architecture)** — decompose the image into *global* per-axis
   labels (`{color, garment, scene}`, one each per image). Better on single-attribute/scene queries,
   but storing colour *globally* means it still can't say *which* garment is blue — so it fails the
   swap just like CLIP.
3. **v2 — G-ADR (this repo)** — the fix is one idea: measure each attribute from its **own garment's
   region**, so binding is structural. This is what takes the swap test from chance to 100%.

The `v1 → v2` shortcomings-and-fixes analysis (what my own testing exposed in ADR-global) is the
opening of [`fashion.md`](fashion.md).

---

## Architecture (two decoupled pipelines over a shared contract)

```
shared/schema.py        SINGLE SOURCE OF TRUTH — axis names, query schema, garment filter
shared/axis_encoding.py how discrete axes become vectors (vocabulary distributions)

indexer/  (Part A: corpus -> Qdrant)
  regions.py        garment REGIONS: SegFormer clothes-parser (ships) | GT masks (oracle)
  attributes.py     per-region attributes: soft-label dists, Lab pixel colour, VQA fallback
  vocab.py          per-axis label vocabularies + prompts
  build_index.py    writes glance_slots (per garment) + glance_images (per image)

retriever/  (Part B: query -> ranked images)
  query_parser.py   Qwen3-4B-Instruct -> structured query (garments[] + scene + vibe)
  search.py         stage-1 ANN candidates -> stage-2 Hungarian binding + rerank
  cli.py            python -m retriever.cli "a red tie and a white shirt" --k 10
```

**The key idea — slots.** Attributes are extracted **per garment region** and stored one
point per garment (a *slot*), so `(garment, colour, pattern)` are bound together by
construction. `colour=red` is tied to the tie because it was measured from the tie's
pixels. Scene and vibe have no garment to bind to, so they live at image level. Two axis
*types*, one contract — see `shared/schema.py`.

**Why discrete axes are stored as distributions, not text embeddings.** Routing colour
through a sentence embedder put "red" and "white" 0.63 apart — too close to bind. Each
slot axis is instead its probability distribution over the axis vocabulary, so matching
"red" measures *how red the garment is* (red-vs-white cosine drops to 0.03). Out-of-vocab
query words ("crimson") soft-map to their nearest neighbours, keeping zero-shot behaviour.
See `shared/axis_encoding.py`.

**Two-stage retrieval.** Stage 1 is a handful of ANN lookups (scales to 1M images).
Stage 2 solves a **Hungarian assignment** between query garments and image slots, so each
colour is scored against *its own* garment; a requested garment that isn't present is
penalised, not silently dropped. This is what makes the swap unrankable by word order.

---

## Dataset (composed, and documented — `eval/` + fashion.md §2)

A **Phase-0 audit gate** ran before any modelling and changed the plan twice:

1. **Fashionpedia has no colour annotations** — colour is derived from segmentation-mask
   pixels (CIELAB histogram) and reported as a pseudo-label, not gold.
2. **Fashionpedia scenes collapse** — 77% runway/studio, **5 office images total**. Three
   of the five eval queries would have had nothing to retrieve.

So the corpus is **composed and documented**:

| Source | Images | Provides |
|---|---|---|
| Fashionpedia val | 1,158 | garments (4.7/image, instance masks, pattern GT) |
| COCO supplement | 1,117 | scenes: office 322 · street 301 · park 249 · home 245 |
| **Total** | **2,275** | 5,190 garment slots |

The COCO supplement is built cheaply and precisely (`eval/build_coco_supplement.py` +
`eval/fetch_coco_images.py`): filter COCO *annotations* for people-in-scene, download only
those, then keep an image only where a human caption **and** CLIP agree on the scene.

---

## Setup & run

```bash
# environment: PyTorch + transformers + qdrant-client + sentence-transformers +
#   scikit-image + scipy + pycocotools  (models auto-download from HF on first use)

# 1. data -> data/ : Fashionpedia val images + instances_attributes_val2020.json,
#    then build the COCO scene supplement:
python eval/build_coco_supplement.py     # caption prefilter (no GPU)
python eval/fetch_coco_images.py         # download + CLIP scene verification (GPU)

# 2. build the index (GPU, ~10 min for 2,275 images)
python -m indexer.build_index                                   # ships: detector regions
python -m indexer.build_index --region fashionpedia_gt \        # oracle, for the ablation
        --collection-suffix _gt

# 3. query
python -m retriever.cli "a red tie and a white shirt in a formal setting" --k 10

# 4. evaluate
python eval/swap_test.py     # G-ADR compositional binding: 100% on 1,259 pairs
python -m eval.baselines     # vanilla CLIP + FashionCLIP on the SAME pairs
```

All thresholds, weights, and model names live in `indexer/config.yaml` and
`retriever/config.yaml` — logic is separated from data.

---

## Results

**Compositional swap test** (does the model bind colour to garment? chance = 50%):

| Model | Swap accuracy | Mean margin (true − swap) |
|---|---|---|
| vanilla CLIP (ViT-B/32) | 57.5% | +0.002 |
| FashionCLIP | 61.2% | +0.003 |
| **Glance (G-ADR)** | **100.0%** (1,259/1,259) | **+0.655** |

Both CLIP baselines sit near chance with a near-zero margin — the quantitative signature
of bag-of-words: the swapped caption is essentially the same vector. Glance separates
every pair, with a margin ~300× larger, because binding is structural (per-garment), not
pooled.

The five assignment queries and their parsed structure are handled natively, including
the two-garment Query 5 that a flat schema cannot even express. See `fashion.md` for the
full design rationale and the v1→v2 exposure analysis.

## Honest limitations
- Colour labels are **pixel-derived pseudo-labels** (Fashionpedia has no colour GT); the
  swap test measures the *binding mechanism* given extraction — extraction accuracy is a
  separate number (hand-labelling, future work).
- The `pattern` axis is noisy (CLIP is weak at fabric pattern) and kept dormant — no eval
  query uses it.
- The ships-detector (SegFormer) has no `tie` class; a GT-vs-detector ablation prices this.
- Scene labels on the COCO supplement are CLIP-verified, not gold.
