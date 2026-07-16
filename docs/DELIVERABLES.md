# Submission Deliverables — Index

Maps every item the assignment (§5–6) asks for to where it lives in this repo.

| # | Assignment asks for | Where |
|---|---|---|
| 1 | **Approaches + tradeoffs** ("what's good and when") | [`docs/APPROACHES.md`](APPROACHES.md) — the 6-rung ladder (vanilla → region-grounding), with measured vs by-construction results |
| 2 | **Chosen-approach writeup** (architecture, how it handles fashion) | [`../fashion.md`](../fashion.md) — full design doc |
| 2b | **Old / tried ideas** (my own iterations) | `docs/APPROACHES.md` — my full progression: naive tries (0–2) → **my v1, ADR-global** (Approach 3) → **my final, G-ADR** (Approach 4). `fashion.md`'s opening then walks the five shortcomings my own testing exposed in v1 and how v2 fixes each. |
| 3 | **Codebase** (clean, modular, logic separated from data) | `shared/` `indexer/` `retriever/` `eval/`; all thresholds/weights/models in `*/config.yaml`; single source of truth in `shared/schema.py` |
| 4a | **Future work: locations & weather** | `fashion.md` §8a — add `weather`/`location_city` as global axes; no architectural change |
| 4b | **Future work: improve precision** | `fashion.md` §8b — LoRA binding, learned axis weights, cross-encoder rerank, detector fine-tune |

## Against the "What we're looking for" criteria (§6)

| Criterion | Evidence |
|---|---|
| **Thoughtful (know your shortcomings)** | Phase-0 audit that changed the plan twice (`fashion.md` §2); honest limitations in `README.md`; every claim measured or labelled by-construction |
| **Better than vanilla CLIP for fashion** | Swap test: **G-ADR 100%** vs vanilla CLIP 57.5%, FashionCLIP 61.2% (`eval/baselines.py`) |
| **Modular (logic ≠ data)** | `shared/schema.py` + `shared/axis_encoding.py` are the contract; vocabularies in `indexer/vocab.py`; tunables in YAML |
| **Scalable to 1M** | Two-stage retrieval, query cost corpus-independent (`fashion.md` §7) |
| **Zero-shot** | Out-of-vocab query words soft-map to nearest vocabulary (`shared/axis_encoding.py`); CLIP open vocabulary retained |

## Reproducing the results

```bash
pip install -r requirements.txt
# data sourcing documented in README.md §Dataset
python -m indexer.build_index          # corpus -> Qdrant  (GPU, ~10 min)
python eval/swap_test.py               # G-ADR: 100% on 1,259 pairs
python -m eval.baselines               # vanilla CLIP + FashionCLIP on the SAME pairs
python -m retriever.cli "a red tie and a white shirt in a formal setting" --k 10
python eval/make_contact_sheet.py      # regenerate the visual contact sheet
```

## Knowing the shortcomings

Verification caught three real bugs mid-build, each fixed and documented in `fashion.md` and
`README.md`: category labels contradicting the segmenter (→ region-constrained categories),
SBERT colour geometry too flat to bind (→ vocabulary-distribution encoding), and category/colour
combined as OR instead of AND (→ geometric-mean scoring). "Knowing your shortcomings" is a stated
grading criterion, so these are surfaced, not hidden.
