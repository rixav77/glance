"""Per-axis label vocabularies and their CLIP prompt templates.

Data, not logic. Adding an axis (weather, location_city -- fashion.md sec.8) means
adding an entry here and a key to the query schema. Nothing else changes: that
extensibility is the whole argument for the axis design.

Two vocabularies carry ground truth from Fashionpedia and are therefore aligned to it
deliberately rather than invented:
  - CATEGORY: the garment supercategories/names Fashionpedia annotates (`tie` included,
    which is what makes eval Query 5 expressible).
  - PATTERN:  Fashionpedia's 19 `textile pattern` attributes (Phase 0, Finding 1 -- the
    one axis that DOES have real labels).

COLOR has no ground truth anywhere in Fashionpedia (Phase 0, Finding 1), so this list is
ours, and colour is extracted primarily from mask pixels rather than from CLIP (attributes.py).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# category -- aligned to Fashionpedia's garment categories
# --------------------------------------------------------------------------
CATEGORY = [
    # upperbody
    "shirt", "blouse", "t-shirt", "sweatshirt", "hoodie", "sweater", "cardigan",
    "blazer", "jacket", "vest", "top",
    # lowerbody
    "pants", "jeans", "shorts", "skirt",
    # wholebody
    "coat", "raincoat", "dress", "jumpsuit", "cape",
    # neck / waist / accessories
    "tie", "scarf", "belt",
    # head / feet / carried
    "hat", "glasses", "shoe", "boot", "bag",
]
CATEGORY_PROMPT = "a photo of a {}"

# --------------------------------------------------------------------------
# color -- no GT in Fashionpedia; extracted mainly from mask pixels
# --------------------------------------------------------------------------
COLOR = [
    "red", "orange", "yellow", "green", "blue", "navy", "purple", "pink",
    "brown", "beige", "black", "white", "grey", "cream", "gold", "silver",
    "maroon", "teal",
]
COLOR_PROMPT = "a photo of a {} garment"

# sRGB anchors for the pixel-histogram extractor. These get converted to CIELAB and
# matched by perceptual distance -- Lab, not RGB, because RGB distance does not track
# what humans (or captions) call "the same colour".
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red":    (200, 30, 40),    "orange": (230, 120, 30),
    "yellow": (240, 210, 60),   "green":  (60, 140, 70),
    "blue":   (50, 90, 190),    "navy":   (25, 35, 80),
    "purple": (120, 60, 160),   "pink":   (235, 150, 180),
    "brown":  (110, 75, 50),    "beige":  (215, 195, 160),
    "black":  (25, 25, 25),     "white":  (243, 243, 243),
    "grey":   (128, 128, 128),  "cream":  (238, 228, 200),
    "gold":   (198, 160, 60),   "silver": (190, 190, 195),
    "maroon": (110, 30, 45),    "teal":   (40, 130, 130),
}

# Colour words that are NOT hues -- a Lab centroid cannot express these, so CLIP keeps
# a job on the colour axis despite being the weaker signal for hue.
COLOR_MODIFIERS = ["bright", "dark", "light", "pastel", "neon", "muted"]

# --------------------------------------------------------------------------
# pattern -- Fashionpedia's `textile pattern` attributes (real GT)
# --------------------------------------------------------------------------
PATTERN = [
    "plain", "striped", "polka dot", "floral", "checked", "plaid", "camouflage",
    "geometric", "paisley", "houndstooth", "herringbone", "chevron", "argyle",
    "abstract", "graphic print",
]
PATTERN_PROMPT = "a photo of {} fabric"

# --------------------------------------------------------------------------
# scene -- identical to the Phase 0 audit bank (eval/audit_scenes.py), on purpose:
# the corpus was SELECTED with these prompts, so the index must use them too.
# --------------------------------------------------------------------------
SCENE = [
    "office", "urban street", "park", "home",
    "studio", "runway", "beach", "restaurant/cafe",
]
SCENE_PROMPTS = {
    "office": "a photo taken inside an office",
    "urban street": "a photo taken on an urban city street",
    "park": "a photo taken in a park or garden",
    "home": "a photo taken inside a home",
    "studio": "a studio photo on a plain seamless backdrop",
    "runway": "a photo of a fashion runway show",
    "beach": "a photo taken at the beach",
    "restaurant/cafe": "a photo taken inside a restaurant or cafe",
}

# --------------------------------------------------------------------------
# style_vibe -- the "vibe" requirement. Kept separate from scene deliberately:
# they correlate (office <-> formal) but they are not the same question, and
# collapsing them would make "casual weekend outfit for a city walk" unanswerable.
# --------------------------------------------------------------------------
STYLE_VIBE = [
    "formal", "business professional", "casual", "streetwear",
    "sporty", "elegant", "weekend casual", "smart casual",
]
STYLE_PROMPT = "a {} outfit"


# --------------------------------------------------------------------------
# Registry -- what attributes.py iterates over.
# --------------------------------------------------------------------------
def _prompts(labels: list[str], template: str) -> dict[str, str]:
    return {lab: template.format(lab) for lab in labels}


AXIS_VOCAB: dict[str, dict[str, str]] = {
    "category":   _prompts(CATEGORY, CATEGORY_PROMPT),
    "color":      _prompts(COLOR, COLOR_PROMPT),
    "pattern":    _prompts(PATTERN, PATTERN_PROMPT),
    "scene":      SCENE_PROMPTS,
    "style_vibe": _prompts(STYLE_VIBE, STYLE_PROMPT),
}

# VQA questions for the low-confidence fallback (fashion.md sec.3.5). Asked of the
# REGION CROP, not the whole image, so the answer stays bound to the garment.
VQA_QUESTION = {
    "category": "What type of clothing is this?",
    "color": "What color is this clothing?",
    "pattern": "What pattern does this fabric have?",
    "scene": "Where was this photo taken?",
    "style_vibe": "What style of outfit is this?",
}


def labels(axis: str) -> list[str]:
    return list(AXIS_VOCAB[axis])


# --------------------------------------------------------------------------
# Region -> allowed fine categories (Phase 0 smoke-test fix)
# --------------------------------------------------------------------------
# The clothes segmenter reliably identifies the COARSE bucket of a region, but CLIP
# on the crop, if left to choose among all 28 categories, mislabels the coarse type a
# third of the time (a `pants` region called `tie`, a `shoe` region called `sweatshirt`).
# So we let the segmenter fix the bucket and let CLIP only refine WITHIN it. `tie` and
# `scarf` live under `upper-clothes` so that eval Query 5's tie stays reachable despite
# the segmenter having no tie class of its own.
REGION_FINE_CATEGORIES: dict[str, set[str]] = {
    "upper-clothes": {"shirt", "blouse", "t-shirt", "sweatshirt", "hoodie", "sweater",
                      "cardigan", "blazer", "jacket", "vest", "top", "tie", "scarf"},
    "pants": {"pants", "jeans", "shorts"},
    "skirt": {"skirt"},
    "dress": {"dress", "jumpsuit", "coat", "raincoat", "cape"},
    "shoe": {"shoe", "boot"},
    "hat": {"hat"},
    "bag": {"bag"},
    "belt": {"belt"},
    "scarf": {"scarf", "tie"},
    "glasses": {"glasses"},
}


def allowed_category_indices(region_label: str | None) -> list[int] | None:
    """Indices into CATEGORY that a given segmenter region may map to. None = no constraint."""
    if region_label and region_label in REGION_FINE_CATEGORIES:
        allowed = REGION_FINE_CATEGORIES[region_label]
        idx = [i for i, n in enumerate(CATEGORY) if n in allowed]
        return idx or None
    return None
