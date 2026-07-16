# Multimodal Fashion & Context Retrieval — Execution Plan v2

**Architecture: Grounded Axis-Decomposed Retrieval (G-ADR)**
Region-grounded attribute binding + soft-label axis embeddings + two-stage retrieve-and-rerank.

> **What changed from v1 (the ADR-global draft) and why.** v1's axis decomposition is the right skeleton, but as
> specified it did not actually deliver the thing it claimed to deliver. Five exposures, all fixed here:
>
> | # | Exposure in v1 | Fix in v2 |
> |---|---|---|
> | 1 | **Compositional binding was asserted, not built.** The indexer ran *global* CLIP zero-shot color classification and stored **one** color label per image, unbound to any garment. Nothing tied `color=blue` to the shirt rather than the pants. "Red shirt + blue pants" and "blue shirt + red pants" would produce near-identical index entries — the exact failure ADR was supposed to fix. Region-grounding, the component that actually creates binding, was deferred to "future work." | **Region-grounding moves into v1** (§3). Attributes are extracted *per garment region*, so `(garment, color, pattern)` are bound in a slot. This is now the contribution, not a stretch goal. |
> | 2 | **No baseline, no metric.** The assignment grades on "better than vanilla CLIP," but v1 never compared against vanilla CLIP and defined no number. Five hand-eyeballed queries prove nothing. | **A measured evaluation is a first-class deliverable** (§6): vanilla CLIP + FashionCLIP baselines, a labeled relevance set, Recall@k / P@k / nDCG@10, and a purpose-built **compositional swap test** that isolates binding accuracy. |
> | 3 | **Hard labels destroyed the ranking signal.** Each image's color axis collapsed to one of ~18 label strings → one of ~18 distinct MiniLM vectors. Hundreds of images share a byte-identical vector; "search" over that axis is a lookup with massive ties, and only the 0.15 visual term breaks them. v1 said it kept the softmax distribution, then never used it. | **Soft-label expectation embeddings** (§3.3): each axis vector is the probability-weighted mean of its vocabulary's label embeddings. Graded, tie-free, same storage cost, no extra models. |
> | 4 | **Dataset fit was assumed.** The eval queries need offices, park benches, and city streets. Fashionpedia is largely runway / product / street-style photography. If `scene` is near-degenerate, three of the five queries have nothing to retrieve. | **Phase 0 audit — now RUN, and it fired** (§2). Fashionpedia is 77% runway+studio with **5 office images**. Corpus is Fashionpedia + a CLIP-verified COCO scene supplement. |
> | 5 | **Two hand-waves.** The BLIP fallback said "extract the axis value from the caption" without saying how; the 0.35 confidence threshold was asserted, not derived. | Fallback is a **VQA call on the region crop** with a constrained answer set (§3.5); the threshold is **calibrated** on a labeled subset, using normalized entropy rather than a raw top-1 score. |
>
> Also folded in: v1's §4.4 punted multi-garment queries ("red tie AND white shirt") to future work.
> With slots, they're handled natively in v1 by bipartite assignment (§4.3). Query 5 stops being the
> known-broken case and becomes the case the architecture is *built* for.
>
> **And one exposure this document did not anticipate, which Phase 0 caught anyway:**
> **Fashionpedia has no color annotations whatsoever.** Three sections of the first v2 draft depended on
> them. Color is now derived from segmentation masks and treated as a measured pseudo-label (§3.4).
> That the gate caught an error in the very plan that specified it is the argument for running gates
> before pipelines.

---

## 1. Problem recap

Retrieve fashion images from a natural-language query, handling:

- single attributes (color, garment type)
- context / location ("in a modern office")
- **compositional binding** ("red shirt *and* blue pants", not the reverse)
- style / vibe inference ("casual weekend outfit")

Vanilla CLIP fails the compositional cases because it pools one embedding per image and one per text.
It is well documented to behave close to a bag-of-words: attributes get *blended* into the global vector
rather than *bound* to the object they modify.

**Core idea.** Attributes are only ever extracted from, and compared within, the spatial region of the
garment they belong to. Binding is enforced by *construction* at index time — never asked of a pooled
embedding, and never patched up during score aggregation.

The distinction that v1 missed, stated plainly:

> Decomposing an image into independent axes is **not** the same as binding attributes to objects.
> `{garment: shirt, color: blue}` extracted globally still cannot tell you the *shirt* was blue.
> Only region-grounding does that.

---

## 2. Phase 0 — dataset audit: RUN, and it changed the plan

**Status: complete.** Corpus = Fashionpedia val (1,158 images, instance masks) + a COCO
scene supplement. Two findings, both of which invalidated assumptions in the first draft of this
document. This is the section to point at when the write-up claims the solution is "thoughtful":
the gate was built to be able to fail, and it did.

*Reproduce with `eval/audit_annotations.py` and `eval/audit_scenes.py` (Slurm job 1974).*

### Finding 1 — Fashionpedia has **no color annotations**. At all.

The 294 attributes are `nickname` (153), `silhouette`, `neckline type`, `textile finishing`,
`textile pattern`, `length`, `opening type`, `material`, `waistline`, `animal`, `leather`. There is
**no color supercategory**. An earlier draft of this plan leaned on color annotations in three separate
places — tuning λ (§3.4), mining swap pairs (§6), generating LoRA hard negatives (§7). All three were
building on something that does not exist.

**Consequence, and it is a real one:** color ground truth must be *derived*, not read. The Lab-space
masked-pixel histogram of §3.4 is promoted from a supporting correction term to the **primary source
of color labels**. This is defensible — masked pixel statistics are substantially more reliable at color
than CLIP is, which is why they were in the plan already — but it is a pseudo-label, and it will be
reported as one: **hand-verify ~100 slots and publish the pseudo-label accuracy** rather than passing
derived colors off as gold. A swap test built on unvalidated pseudo-labels would be measuring the
color extractor, not the retriever.

**Free win:** the 19 `textile pattern` attributes (plain, stripe, dot, floral, check, camouflage,
houndstooth, paisley, …) mean the `pattern` axis *does* have real ground truth.

### Finding 2 — the garment side is strong, but the categories are not all garments.

**1,093 / 1,158 images (94%) carry ≥2 main garments**, mean 4.7 per image. The slot architecture has
ample material to bind, and the swap-test population is large. Critically, **`tie` is its own category**,
so Query 5 ("a red tie and a white shirt") is directly supported by the schema.

But the 46 categories mix garments with garment *parts* — `sleeve`, `collar`, `pocket`, `neckline`,
`hood`, `lapel` are all annotated instances, and `sleeve` alone is the second most common. Slots must
be **filtered by supercategory**: keep `upperbody`, `lowerbody`, `wholebody`, `neck`, `waist`,
`legs and feet`, `head`, `others`; drop `garment parts`, `closures`, `decorations`. Skipping this filter
would fill the index with slots for sleeves and zippers and quietly wreck the assignment step in §4.3.

### Finding 3 — the scene axis on Fashionpedia is **unusable**, exactly as feared.

CLIP zero-shot scene classification over all 1,158 val images:

| Scene | Images | Share |
|---|---|---|
| runway | 711 | 61.4% |
| studio | 184 | 15.9% |
| park | 109 | 9.4% |
| urban street | 109 | 9.4% |
| beach | 26 | 2.2% |
| home | 8 | 0.7% |
| restaurant/cafe | 6 | 0.5% |
| **office** | **5** | **0.4%** |

Normalized entropy **0.58**; **77.3% runway + studio**. There are **five office images in the entire
corpus** and eight home images. Query 2 ("professional business attire inside a modern office") has
essentially nothing to retrieve — and no retrieval architecture, however clever, can return an image
the corpus does not contain. Park and street survive at ~109 each: thin, but workable.

Note what this would have looked like without the gate: a fully-built pipeline returning five
inexplicable contact sheets, and a long, wrong hunt through the model for a bug that was in the data.

### The fix — a scene supplement from COCO

Fashionpedia stays as the **garment-rich, binding-rich core** (real masks, real pattern GT, 4.7
garments/image). We add real photographs of **people in real scenes** from COCO, which carries exactly
what Fashionpedia lacks: offices, kitchens, living rooms, sidewalks, parks. The assignment explicitly
permits *sourcing or simulating* the dataset, so a deliberately composed corpus is legitimate — provided
its composition is documented, which is what this section is for.

Selection is a two-stage funnel, built to be cheap and to be *precise*:

1. **Caption prefilter** (`build_coco_supplement.py`, no GPU, no bulk download). COCO train2017's images
   are 18 GB, so we never fetch them wholesale. Instead we filter *annotations*: images containing a
   person instance of area > 20,000 px (small background figures show no clothing), whose captions match
   a scene word list. Yields 1,600 candidates — including **610 office**, against Fashionpedia's 5.
2. **CLIP verification** (`fetch_coco_images.py`, GPU). Download only those candidates, then classify each
   against the *same* scene prompt bank the indexer uses, and keep an image only where **the human
   captioner and CLIP agree** on the scene, above a confidence floor of 0.30. Rejected images are deleted.

The agreement requirement is the point. It means every supplement image is one that both a human and the
model call an office, so the `scene` axis has a population it can actually retrieve from — rather than a
population that merely *claims* to be offices in a caption.

**Reported honestly in the write-up:** the corpus is composed, not found; the scene labels on the
supplement are CLIP-verified rather than gold; and per-scene support is printed next to every metric in
§6, so a strong number on a thin scene cannot masquerade as a real result.

---

## 3. Part A — the indexer

### 3.1 The representation

Each image becomes **a set of garment slots plus a small set of global axes**. Slots are the whole
point: they are what make an attribute *belong* to something.

```
Image
├── slots: [                      # one per detected garment region — the binding unit
│     { region, category, color, pattern },   # e.g. {shirt, white, solid}
│     { region, category, color, pattern },   # e.g. {tie,   red,   solid}
│     ...
│   ]
├── scene:        <soft-label embedding>      # genuinely image-level
├── style_vibe:   <soft-label embedding>      # genuinely image-level
└── visual_global:<CLIP image embedding>      # fine detail the discrete axes throw away
```

`scene` and `style_vibe` stay global because they genuinely *are* properties of the whole image —
there is no region to bind them to. `color` and `pattern` never do, because they always modify a
garment. That asymmetry is the entire fix for Exposure 1.

**Design rule (carried over from v1, still correct and still load-bearing):** an axis absent from the
query gets **weight 0**. That's what lets "professional business attire inside a modern office" —
no color, no pattern — match on `scene` + `style_vibe` alone instead of being diluted by axes it
never mentioned.

### 3.2 Getting the regions

Two paths; **build both**, because the comparison between them is itself a result worth reporting.

- **Detector path (default, and the one that ships).** An open-vocabulary garment detector — a
  Fashionpedia-trained Mask R-CNN, or Grounding-DINO / YOLO-World prompted with the garment
  category list. Works on *any* image, including the supplemented scene-rich ones that carry no
  annotations. This is the honest system: it does not depend on ground-truth labels existing.
- **Oracle path (evaluation only).** Fashionpedia ships instance segmentation masks. Index a subset
  using the ground-truth masks instead of the detector.

Running both and reporting the delta gives a clean **ablation that prices detector error**: how much
retrieval quality is lost to imperfect grounding versus perfect grounding. That is exactly the kind of
"understand your architecture's shortcomings" analysis the assignment says it is looking for, and it
costs one extra index build.

*Degenerate fallback if detection proves flaky:* person-detect, then split the box into upper / lower /
foot thirds. Crude, but it still binds — and crude binding beats no binding, which is what v1 had.

### 3.3 Per-slot and per-axis attribute extraction — soft labels, not hard ones

For each region crop, and for each global axis, run CLIP zero-shot against a per-axis prompt bank
(`vocab.py`):

| Axis | Prompt template | Vocab size |
|---|---|---|
| `category` | `"a photo of a {garment}"` | Fashionpedia's 46 categories |
| `color` | `"a photo of a {color} garment"` | ~18 colors |
| `pattern` | `"a photo of {pattern} fabric"` | ~8 patterns |
| `scene` | `"a photo taken in a {scene}"` | ~8 scenes |
| `style_vibe` | `"a {style} outfit"` | ~8 styles |

Then — **and this is the fix for Exposure 3** — do *not* take the argmax and embed the winning string.
Store the **soft-label expectation embedding**:

```
p(v | region) = softmax_v ( CLIP_img(region) · CLIP_txt(prompt(v)) / τ )

e_axis(image) = normalize( Σ_v  p(v | region) · SBERT(v) )
```

where `SBERT(·)` is the sentence encoder (`all-MiniLM-L6-v2`) applied to the label. At query time the
query side is just `SBERT(extracted_phrase)`, and matching is plain cosine — so this drops straight
into the same Qdrant named-vector machinery with zero extra storage and no extra model.

Why it matters, concretely:

- **v1:** a navy blazer at p=0.34 and a black blazer at p=0.33 collapse to *byte-identical* vectors.
  Every "navy" query returns them tied, and a 0.15-weight visual term arbitrates. Ranking is noise.
- **v2:** their vectors differ in proportion to the model's actual confidence. A query for "navy" ranks
  the confidently-navy garment above the ambiguous one. The graded signal CLIP produced — and
  which v1 computed, stored, and then ignored — is finally *used*.

It also preserves paraphrase resilience for free: because the vector lives in sentence-encoder space,
a "crimson" query still lands near a red-dominant distribution, and "workplace" near "office."

### 3.4 Color: derived from pixels, because the dataset has none

**Phase 0 finding 1 makes this section load-bearing rather than optional.** Fashionpedia annotates no
colors, so there is no color label to read — it has to be computed. Fortunately this is the one attribute
where that's an *advantage*: CLIP is genuinely mediocre at color (it is trained on captions, where color
words are sparse and inconsistent), whereas **masked pixel statistics are near-exact at it**.

Per slot, compute a **Lab-space color histogram over the garment mask** — the mask excludes background,
which is precisely the thing that poisons global color classification — and map it to the color vocabulary
by nearest centroid in a perceptually-uniform space. Blend with CLIP's opinion:

```
p_color(v) = λ · p_CLIP(v) + (1 − λ) · p_pixel(v)
```

with the pixel term dominant (**λ ≈ 0.3**, i.e. weighted *toward* pixels — the opposite of what you'd
choose if you trusted CLIP here). CLIP still earns its keep on color words that aren't a hue at all
("bright", "pastel", "dark", "neon"), which a histogram centroid cannot express.

**These are pseudo-labels, and the plan treats them as such.** With no gold color, λ cannot be tuned
against ground truth, and the §6 swap test would otherwise be built on unvalidated derived labels —
which would silently make it a test of the color extractor rather than of the retriever. So:

- **Hand-label ~100 garment slots** (a couple of hours) → a small gold set.
- Tune λ on it, and **report pseudo-label accuracy**: CLIP-only vs. pixel-only vs. blended.
- Every downstream claim that depends on color is then bounded by a number we published, instead of
  resting on an assumption we never checked.

This is the honest cost of Finding 1, and stating it plainly is worth more than a swap-test score with
an unexamined foundation.

### 3.5 Low-confidence fallback (specified, not hand-waved)

v1 said "run BLIP and extract the axis value from the caption" without saying how, and asserted a 0.35
threshold from nowhere. Both are now pinned down:

- **Trigger.** Fire when the axis distribution is *uninformative*, measured by **normalized entropy**
  `H(p) / log|V| > θ` — not by a raw top-1 probability, which is not comparable across vocabularies
  of different sizes (a top score of 0.35 means something very different over 8 labels than over 46).
- **Threshold.** `θ` is **calibrated**, not asserted: label ~100 images by hand, sweep `θ`, pick the knee
  where fallback fires on genuinely-wrong predictions without firing on correct ones. Report the curve.
- **Mechanism.** **BLIP-VQA on the region crop**, not free-form captioning: ask
  `"What color is this garment?"` / `"What type of garment is this?"` and constrain decoding to the
  axis vocabulary, with an escape hatch for genuinely out-of-vocab answers ("chartreuse"), whose raw
  string is embedded with SBERT directly. That escape hatch is what preserves **zero-shot capability
  beyond the fixed label set** — the assignment's fourth grading criterion.

### 3.6 Storage

**Qdrant**, two collections (per the assignment's "pick the easiest vector DB, don't rewrite one" guidance):

- `slots` — one point **per garment region**. Named vectors: `category`, `color`, `pattern`.
  Payload: `{image_id, bbox, confidences, raw_labels}`.
- `images` — one point **per image**. Named vectors: `scene`, `style_vibe`, `visual_global`.
  Payload: `{image_path, slot_ids, scene/style confidences}`.

One point per *slot* rather than per *image* is what lets variable-length garment sets live in a
fixed-schema vector DB without any custom fusion code — and it makes stage-1 candidate generation
a single ANN call (§4.2).

### 3.7 What still doesn't get built

- No custom ANN. Qdrant's HNSW is fine.
- No fine-tuning in v1. (LoRA on hard negatives stays future work — §7.)
- No LLM captioning of every image. Fallback only, on low-entropy-failure axes.

---

## 4. Part B — the retriever

### 4.1 Query parsing → a structured query, not a flat dict

The parser (small instruction-tuned LLM, fixed-schema JSON output) now emits **a list of garment
constraints plus global constraints** — mirroring the index's shape exactly, which is the whole reason
matching stays honest:

```jsonc
// "A red tie and a white shirt in a formal setting"
{
  "garments": [
    {"category": "tie",   "color": "red",   "pattern": null},
    {"category": "shirt", "color": "white", "pattern": null}
  ],
  "scene": null,
  "style_vibe": "formal"
}
```

Note what v1's flat schema could not express: two garments, each with its *own* color. That
representational gap is why v1 had to declare Query 5 a known limitation. Fix the schema, and the
limitation evaporates.

`shared/schema.py` remains the single source of truth for axis names and vocabularies, imported by
both pipelines, so they cannot silently drift. (This is the assignment's "is your logic separated from
your data" criterion, and it's the one thing v1 got unambiguously right.)

### 4.2 Two-stage retrieval

**Stage 1 — candidate generation (ANN, sub-linear).** For each query garment constraint, ANN-search
the `slots` collection on its most selective axis; for global constraints, ANN-search `images`. Union the
resulting `image_id`s → a few hundred candidates. *This is the stage that has to scale to 1M images,
and it's a small fixed number of HNSW lookups regardless of corpus size.*

**Stage 2 — exact rerank with binding (O(candidates), not O(corpus)).** For each candidate image,
solve the assignment problem between query garment constraints and the image's slots.

### 4.3 Binding via bipartite assignment — Query 5 handled natively

Build a cost matrix between the *q* query garment constraints and the *m* slots of a candidate image:

```
M[i][j] = mean over specified attrs a of  cos( SBERT(q_i.a),  e_a(slot_j) )
```

(attributes the query left `null` simply don't enter the mean — the zero-weight rule, applied *inside*
a slot). Solve with **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`; *q* ≤ 5, *m* ≤ 10,
so cost is negligible). Final score:

```
score(image) =  w_g · (mean of matched slot scores)          # bound garment attributes
              + w_s · cos( SBERT(q.scene),      e_scene(img) )      # if present
              + w_v · cos( SBERT(q.style_vibe), e_style(img) )      # if present
              + w_c · cos( CLIP_txt(full_query), e_visual_global(img) )   # small, w_c ≈ 0.15
```

Unmatched query garments (the image has no tie at all) incur an explicit miss penalty rather than
being silently dropped — otherwise an image with one white shirt and no tie scores *identically* to one
with both, which would quietly reintroduce the bag-of-words failure through the back door.

**This is where the architecture earns its keep.** For "red tie **and** white shirt":

- **Vanilla CLIP** pools `red + tie + white + shirt` into one vector and cannot distinguish it from a
  white tie with a red shirt. Both are bags of the same words.
- **v1 ADR** stores one global color per image — call it `red` — and one global garment — `shirt`.
  It has *no idea* which garment was red. It would have matched the swapped image just as happily.
- **v2 G-ADR** matches `{tie, red}` against a slot whose attributes were extracted *from the tie's
  pixels*, and `{shirt, white}` against a slot extracted *from the shirt's pixels*. The swapped image
  scores strictly lower, because its tie-slot is white and its shirt-slot is red. **The binding is
  structural.** It cannot be fooled by word order, because word order was never what it was reading.

### 4.4 The five evaluation queries

| # | Query | Parsed structure | What it tests, and why v2 handles it |
|---|---|---|---|
| 1 | "A person in a bright yellow raincoat" | 1 garment `{raincoat, yellow}` | Sanity baseline. Vanilla CLIP also passes this; v2 should too, and the masked-pixel color term (§3.4) should make it *cleaner*. |
| 2 | "Professional business attire inside a modern office" | `scene=office`, `style_vibe=professional` | **Zero-weight rule.** No garment, no color → those axes contribute nothing rather than diluting the match. |
| 3 | "Someone wearing a blue shirt sitting on a park bench" | 1 garment `{shirt, blue}` + `scene=park` | **Binding + context together.** The `blue` is compared only against the *shirt slot's* color — an image with blue jeans in a park cannot masquerade as a match. This is precisely the case v1 claimed to solve and did not. |
| 4 | "Casual weekend outfit for a city walk" | `style_vibe=casual`, `scene=urban street` | **Pure vibe + scene, no garment.** Tests whether the global axes can carry a query alone. Also the query most at risk from Phase 0's dataset finding — worthless if the corpus has no streets. |
| 5 | "A red tie and a white shirt in a formal setting" | 2 garments `{tie, red}`, `{shirt, white}` + `style_vibe=formal` | **The compositional test, and now the showcase rather than the known bug.** Hungarian assignment binds each color to its own garment. §6's swap test measures exactly this. |

---

## 5. Repo structure

```
repo/
├── indexer/
│   ├── detect_regions.py     # garment detection/segmentation → slots (detector | GT-mask oracle)
│   ├── extract_attributes.py # per-slot & per-axis CLIP soft-label embeddings + pixel color + VQA fallback
│   ├── vocab.py              # per-axis prompt banks / label lists
│   ├── build_index.py        # writes slot + image points into Qdrant
│   └── config.yaml           # thresholds, λ, weights, model names
├── retriever/
│   ├── query_parser.py       # LLM → structured query (garment list + global axes)
│   ├── search.py             # stage-1 ANN candidates → stage-2 Hungarian rerank
│   └── cli.py                # python cli.py "a red tie and a white shirt" --k 10
├── shared/
│   └── schema.py             # SINGLE SOURCE OF TRUTH: axis names, vocabularies, query schema
├── eval/
│   ├── build_eval_set.py     # labeled relevance set + compositional swap pairs
│   ├── baselines.py          # vanilla CLIP, FashionCLIP, ADR-global (= v1), G-ADR (= v2)
│   ├── run_metrics.py        # Recall@k, P@k, nDCG@10, swap accuracy → results table
│   └── run_eval_queries.py   # the 5 assignment queries → top-k contact sheet
├── data/                     # not committed; sourcing documented in README
└── README.md
```

---

## 6. Evaluation — the deliverable v1 didn't have

The assignment says the solution must be **better than vanilla CLIP**. That is a *comparative,
empirical* claim. It cannot be settled by looking at five contact sheets, and a write-up that asserts it
without a number is just a hypothesis with confidence. This section is not optional polish — it is
what converts the whole project from a plausible design into a demonstrated result.

**Systems compared** (each one isolates exactly one design decision):

| System | Isolates |
|---|---|
| Vanilla CLIP ViT-B/32, cosine on pooled embeddings | The baseline the assignment names |
| FashionCLIP (fashion-domain CLIP) | *Is a domain-tuned backbone all you actually needed?* An honest, and genuinely uncomfortable, question worth answering |
| **ADR-global** — v1 as written: global axes, hard labels | The cost of *not* grounding, and of hard labels |
| **G-ADR** — v2: grounded slots, soft labels, Hungarian | The full system |
| G-ADR + GT masks (oracle) | The price of detector error (§3.2) |

**Relevance set.** ~25–30 queries spanning the five assignment archetypes. Pool the top-20 from every
system, judge each `(query, image)` pair for relevance once, reuse across all systems (standard pooled
TREC-style judging — this is what makes the comparison fair, since no system gets to be graded on
results the others never surfaced). ~500 judgments, a couple of hours.

**Metrics.** Recall@10, Precision@5, nDCG@10 — plus per-scene support, so a strong number on a
sparsely-populated scene can't hide.

**The compositional swap test — the metric that actually matters here.** Everything else is table
stakes; this is the one that tests the architecture's central claim.

Take images with ≥2 main garments of *different* derived colors (e.g. one that genuinely contains a red
top and blue trousers). Phase 0 confirmed the population is large: **94% of val images carry ≥2 main
garments**, mean 4.7 each. Colors come from the §3.4 pixel extractor, **not** from annotations — there
are none — so the pairs are validated against the hand-labeled gold slots before use, and the swap-test
result is reported alongside the pseudo-label accuracy that bounds it. For each image, construct a
matched query pair:

- **True query:** "a red shirt and blue pants"
- **Swapped query:** "a blue shirt and red pants"

A system with real attribute binding ranks the correct image higher for the true query than for the
swapped one. A bag-of-words system scores them ~identically — it *cannot tell the two queries apart*.

Report **binding accuracy** = fraction of pairs where `rank(true) < rank(swapped)`, and the mean score
margin between them. Predicted, and worth stating up front as a falsifiable claim:

- Vanilla CLIP: **~50%** — chance. It is functionally blind to word order here.
- ADR-global (v1): **~50–60%.** *This is the number that proves Exposure 1 was real.* v1 would have
  shipped believing it had solved compositionality while scoring at chance on the only test of it.
- G-ADR (v2): **substantially above chance**, bounded by detector and color-extraction quality.

If G-ADR does *not* clear ADR-global on this metric, the architecture's central claim is false and the
write-up must say so. Building the test that can falsify your own thesis — and reporting the result
either way — is the "thoughtful solution" the assignment is grading. Design the experiment so it
*could* embarrass you; that's the only kind whose passing means anything.

---

## 7. Scalability (1M images)

- **Indexing is embarrassingly parallel** — one image at a time, zero cross-image dependency. Trivially
  shardable. Detection is the new cost driver (~an extra GPU-pass per image); at 1M that's a batch job,
  not an architectural problem.
- **Storage:** ~4 slots/image × 3 × 384-d + 2 × 384-d + 512-d ≈ 6 KB/image → ~6 GB at 1M. Comfortable
  for Qdrant, and the slot collection sharding by `image_id` is standard.
- **Query cost is corpus-independent by construction.** Stage 1 is a fixed handful of HNSW lookups
  (sub-linear). Stage 2 — the expensive, exact, Hungarian part — runs only over the few hundred
  candidates stage 1 returned, **never over the corpus**. This retrieve-then-rerank split is precisely
  what makes an expensive, high-precision scorer affordable at scale, and it's the honest answer to the
  assignment's 1M-image question.
- **The LLM parser is the only per-query external latency**, and it is independent of dataset size.
  Cache by exact query string.

---

## 8. Future work

**a. Locations (cities, places) and weather.** Add `weather` and `location_city` as two more **global**
axes — same prompt-bank zero-shot extraction at index time (`"a photo taken on a {weather} day"`),
same key in the parser's JSON schema, same term in the score. No architectural change; they slot in
beside `scene` and `style_vibe`. That the extension is this boring is the payoff of the axis design, and
the reason to keep the schema in one file. Beyond zero-shot: GPS/EXIF metadata where available, a
geo-landmark classifier for city identification, and — since weather correlates strongly with outerwear —
a learned prior linking the `weather` and `garment` axes rather than treating them as independent.

**b. Improving precision.**

- **Move binding into the representation.** LoRA on CLIP's projection layers, trained with hard negatives
  that *swap attribute bindings*. Note this cannot be sourced from Fashionpedia's attributes as an earlier
  draft assumed — there are no color annotations (Phase 0, Finding 1) — so the negatives would be
  generated from the §3.4 mask-derived colors, and the quality of the fine-tune is therefore capped by the
  pseudo-label accuracy we measure. v2 enforces binding structurally, *outside* the model; this would teach
  the model to do it itself — cheaper at query time and strictly more general.
- **Learn the axis weights.** `w_g, w_s, w_v, w_c` are hand-set. With the §6 relevance set in hand,
  fit them (LambdaMART / a small MLP) and turn aggregation into a lightweight trainable ranker.
- **A cross-encoder final stage.** Stage 2 already reranks a small candidate set — that's exactly the
  budget where a VLM cross-encoder scoring `(query, image)` jointly becomes affordable. Highest-precision
  option available, essentially free architecturally given the two-stage split already exists.
- **Better grounding.** Detector error is the dominant remaining source of loss (and §6's oracle ablation
  will tell you *exactly how much*, which is the point of running it). Fine-tune the detector on
  Fashionpedia masks; handle occlusion and multi-person images explicitly.
- **Query-side expansion.** "Casual weekend outfit" is a *distribution* over garments, not a garment.
  Expand vibe-only queries into likely garment sets via an LLM prior, then match those as soft slot
  constraints.
