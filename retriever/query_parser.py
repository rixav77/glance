"""Query parser: natural language -> ParsedQuery (fashion.md sec.4.1).

Emits the SAME shape as the index: a list of garment constraints, each with its own
colour/pattern, plus image-level scene and style_vibe. That per-garment structure is
the whole reason "a red tie and a white shirt" can be matched honestly -- a flat
{color: red, garment: shirt} dict physically cannot say which garment is red, which is
why the earlier flat design had to declare Query 5 a known limitation.

An instruction-tuned LLM does the decomposition (general: it handles paraphrases and
implicit attributes a keyword matcher would miss). A deterministic vocab-scan fallback
runs if the model is unavailable or emits unparseable JSON, so the pipeline degrades
instead of crashing.
"""

from __future__ import annotations

import json
import re

from indexer import vocab
from shared.schema import GarmentConstraint, ParsedQuery

SYSTEM = """You extract structured search constraints from a fashion image search query.
Return ONLY a JSON object, no prose, with exactly these keys:
  "garments": a list of objects, each {"category":..., "color":..., "pattern":...}
  "scene": one of office/urban street/park/home/beach/restaurant/studio/runway, or null
  "style_vibe": e.g. formal/business professional/casual/streetwear/sporty/elegant, or null

Rules:
- One garment object per distinct clothing item mentioned. Bind each color/pattern to ITS garment.
- Use null for any attribute not mentioned. Do NOT invent attributes.
- If no specific garment is mentioned (only a scene or vibe), "garments" is an empty list.
- category is the clothing type (tie, shirt, dress, raincoat, pants...); color and pattern are its attributes.
- Set "scene" ONLY when a physical LOCATION is explicitly named (office, park, street, home, beach).
  A vibe/formality word is NOT a location: "a formal setting" -> style_vibe="formal", scene=null.
  Never guess a scene from a vibe."""

FEWSHOT = [
    ("A red tie and a white shirt in a formal setting",
     {"garments": [{"category": "tie", "color": "red", "pattern": None},
                   {"category": "shirt", "color": "white", "pattern": None}],
      "scene": None, "style_vibe": "formal"}),
    ("Professional business attire inside a modern office",
     {"garments": [], "scene": "office", "style_vibe": "business professional"}),
    ("Someone wearing a blue shirt sitting on a park bench",
     {"garments": [{"category": "shirt", "color": "blue", "pattern": None}],
      "scene": "park", "style_vibe": None}),
    ("A person in a bright yellow raincoat",
     {"garments": [{"category": "raincoat", "color": "yellow", "pattern": None}],
      "scene": None, "style_vibe": None}),
]


class QueryParser:
    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
                 device: str = "cuda", use_llm: bool = True):
        self.use_llm = use_llm
        self.model = self.tok = None
        if use_llm:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
            self.device = device

    # -- LLM path ---------------------------------------------------------
    def _llm_json(self, query: str) -> dict | None:
        import torch

        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEWSHOT:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": json.dumps(a)})
        msgs.append({"role": "user", "content": query})

        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return _extract_json(gen)

    # -- deterministic fallback ------------------------------------------
    @staticmethod
    def _vocab_scan(query: str) -> dict:
        """No LLM: scan for known vocabulary. Cannot bind colours to specific garments
        as well as the LLM, but never crashes and covers the common single-garment case."""
        q = query.lower()
        cats = [c for c in vocab.CATEGORY if re.search(rf"\b{re.escape(c)}\b", q)]
        colors = [c for c in vocab.COLOR if re.search(rf"\b{re.escape(c)}\b", q)]
        garments = []
        for i, cat in enumerate(cats):
            garments.append({"category": cat,
                             "color": colors[i] if i < len(colors) else None,
                             "pattern": None})
        if not cats and colors:                       # colour but no garment word
            garments.append({"category": None, "color": colors[0], "pattern": None})
        scene = next((s for s in vocab.SCENE if s in q), None)
        vibe = next((v for v in vocab.STYLE_VIBE if v in q), None)
        return {"garments": garments, "scene": scene, "style_vibe": vibe}

    # -- public -----------------------------------------------------------
    def parse(self, query: str) -> ParsedQuery:
        raw = None
        if self.use_llm:
            try:
                raw = self._llm_json(query)
            except Exception as e:
                print(f"[parser] LLM failed ({e}); using vocab-scan fallback")
        if raw is None:
            raw = self._vocab_scan(query)
        return _to_parsed(query, raw)


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _norm(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    return v or None


def _to_parsed(query: str, d: dict) -> ParsedQuery:
    garments = []
    for g in d.get("garments") or []:
        gc = GarmentConstraint(category=_norm(g.get("category")),
                               color=_norm(g.get("color")),
                               pattern=_norm(g.get("pattern")))
        if not gc.is_empty():
            garments.append(gc)
    return ParsedQuery(raw=query, garments=garments,
                       scene=_norm(d.get("scene")), style_vibe=_norm(d.get("style_vibe")))
