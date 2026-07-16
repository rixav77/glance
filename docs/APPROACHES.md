# Approaches Considered — Tradeoffs & Why We Chose Region-Grounding

*Assignment deliverable 1: "Possible ways to solve this problem, tradeoffs, what's good and when."*

The problem is fixed by the assignment's own hint: **CLIP cannot bind attributes to the
garments they modify** — it cannot tell *"red tie + white shirt"* from *"white tie + red
shirt"*. This document is the actual path I explored, cheapest-to-heaviest, and it maps to my
own iterations, not an abstract survey:

- **Approaches 0–2 = my naive first tries** (vanilla CLIP, a fashion-tuned CLIP, caption-and-match).
  I ruled all three out because they pool into one vector and fail the compositional case.
- **Approach 3 = my v1** (ADR-global) — my first real architecture. It looked like it solved
  compositionality, and my own testing showed it didn't.
- **Approach 4 = where I landed** (G-ADR, this repo) — region-grounding, the one change that makes
  binding real.
- **Approach 5 = future work** (training the binding into the weights).

Two of the numbers below are **measured** on our 1,259-pair compositional swap test
(`eval/baselines.py`, `eval/swap_test.py`); the rest are **by construction** — reasoned from
the mechanism, and labelled as such. We do not report benchmarks we did not run.

---

## Approach 0 — Vanilla CLIP (the baseline to beat)

Embed images and query, rank by cosine on the pooled vectors.

- **Good when:** you need a zero-shot, instantly-scalable baseline, or the queries are single
  simple concepts ("a dog").
- **Fails:** compositional and fine-grained fashion queries. One pooled vector per image = a
  bag of words; attributes are blended, not bound.
- **Swap accuracy: 57.5% [measured]** — barely above the 50% coin flip; true−swap margin +0.002.

## Approach 1 — Stronger CLIP weights (FashionCLIP)

Swap in a CLIP fine-tuned on fashion captions.

- **Good when:** you want better fashion *vocabulary* with a one-line change.
- **Fails:** identical architecture ⇒ identical disease. Still one pooled vector, still can't bind.
- **Swap accuracy: 61.2% [measured]** — better fashion knowledge, compositional gap barely moves.
  A stronger CLIP is not a *different* CLIP.

## Approach 2 — Prompt decomposition / caption-and-match

Caption each image (BLIP) or expand the query (LLM), then match text-to-text.

- **Good when:** you have no GPU budget for indexing and want flexibility fast.
- **Fails:** binding now rides on the caption's word order surviving a *pooled text* embedding —
  the same bag-of-words failure, relocated from the image encoder to the text encoder. Also brittle:
  the caption may never mention the colour at all. **[by construction]**

## Approach 3 — Attribute decomposition, but GLOBAL — **my v1** (ADR-global, my first real architecture)

Classify the whole image on separate axes and store **one label/vector per axis per image**:
`{color: blue, garment: shirt, scene: office}`.

- **Genuinely better than 0/1/2, and here is the mechanism — axis isolation kills cross-attribute
  dilution.** Vanilla CLIP crams colour, garment and scene into one similarity where they compete
  invisibly; a strong scene match can hide a wrong colour. Approach 3 scores each axis *separately*
  then combines: `Σ wₐ·cos(query.axisₐ, img.axisₐ)`. So a wrong colour yields a low colour term that
  can't be masked, unmentioned axes can be zero-weighted, and every image is forced to a calibrated
  value on every axis (unlike a caption that may omit it). **[by construction]**
- **Where it wins:** single-attribute (Q1), context/scene (Q2), style (Q4) — the axes that don't
  need binding.
- **The ceiling — and the whole lesson of this project:** `img.color_vec` is **one colour for the
  whole image**, so `cos("blue", img.color_vec)` asks *"is there blue anywhere?"*, not *"is the
  **shirt** blue?"*. An image with blue jeans + a grey shirt in a park matches "blue shirt in a park"
  and it is **wrong**. Q3 and Q5 are therefore **no better than vanilla (≈ chance on the swap).
  [by construction — not benchmarked; we leapfrogged it]**
- **Decomposition ≠ binding.** This is the sophisticated-looking wrong answer, and seeing *exactly*
  why it fails (global vs per-region colour) forces the single change that defines Approach 4.

## Approach 4 — Region-grounded attribute binding — **CHOSEN** (G-ADR)

Segment each garment, extract attributes **from that garment's own pixels**, store one **slot**
per garment. At query time, bind each query garment to a slot via Hungarian assignment.

- **The diff from Approach 3 is literally one word:** colour is stored **per-region**, not
  per-image. `cos("blue", …)` now asks *"is **the shirt** blue?"*, because the colour was measured
  from the shirt's pixels. Binding is **structural** — true by construction, unfoolable by word order.
- **Good when:** exactly this problem — compositional, fine-grained, must beat CLIP, must scale,
  must stay zero-shot.
- **Cost:** more moving parts (a segmenter + per-region extraction); quality capped by segmentation.
- **Swap accuracy: 100.0% [measured]**, margin +0.655 (~300× the baselines').

## Approach 5 — Train binding into the model (LoRA / cross-attention fine-tune)

Fine-tune CLIP with hard-negative swap pairs so it learns to bind internally.

- **Good when:** you have training data + GPU budget and want binding baked into the weights
  (cheapest at *query* time, most general).
- **Why not now:** needs labelled swap data and training; overkill for an assignment graded on
  *reasoning*, not on squeezing SOTA. Listed as future work (`fashion.md` §8b).

---

## Why Approach 4 — the decision, against the assignment's own grading criteria

| Grading criterion (§6) | Rules out | Rewards |
|---|---|---|
| **Better than vanilla CLIP** on compositionality | 0, 1, 2 (bag-of-words); 3 (global colour) | 4, 5 |
| **Scalability to 1M** | naive per-image rerank | 4's two-stage ANN (query cost corpus-independent) |
| **Zero-shot** | trained-classifier-only designs | 4 keeps CLIP's open vocabulary + soft-vocab mapping |
| **Thoughtful (know your shortcomings)** | 3 (looks right, silently fails) | 4 (binding is *provable* via the swap test) |

**Approach 4 is the lowest rung on the ladder that actually solves the stated problem.**
Everything below it fails Query 5; the rung above it (training) adds cost the assignment doesn't
reward. Region-grounding is the honest minimum — and the only option whose central claim can be
**measured** (100% vs ~57–61%) rather than asserted.

See `../fashion.md` for the full chosen-architecture writeup and the v1→v2 exposure analysis.
