"""Render the 5-query results (eval_queries_results.json) as a visual contact sheet.

Embeds each retrieved image as a base64 thumbnail (the Artifact CSP blocks external
hosts, so images must be inlined) and annotates every frame with its BINDING evidence --
which query garment matched which slot, at what strength -- so the page shows not just
what was retrieved but why.
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexer.vocab import COLOR_RGB

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "eval" / "eval_queries_results.json").read_text())
OUT = ROOT / "eval" / "contact_sheet.html"

# vocabulary colour name -> CSS swatch (subject-grounded: colour is the whole point)
SWATCH = {n: f"rgb({r},{g},{b})" for n, (r, g, b) in COLOR_RGB.items()}


def thumb(path: str, long_edge: int = 460, q: int = 82) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = long_edge / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def swatch(color: str) -> str:
    css = SWATCH.get(color, "linear-gradient(135deg,#bbb,#777)")
    ring = "box-shadow:inset 0 0 0 1px rgba(0,0,0,.25)" if color in ("white", "cream") else ""
    return f'<span class="sw" style="background:{css};{ring}"></span>'


def strength_class(sim: float) -> str:
    return "hit" if sim >= 0.5 else ("part" if sim >= 0.25 else "miss")


def chip_garment(g: dict) -> str:
    col, cat = g.get("color"), g.get("category")
    parts = []
    if col:
        parts.append(swatch(col) + html.escape(col))
    if cat:
        parts.append(html.escape(cat))
    return f'<span class="chip">{" ".join(parts)}</span>'


def render_matches(matches: list) -> str:
    if not matches:
        return '<div class="bind none">scene &amp; vibe only — no garment requested</div>'
    rows = []
    for m in matches:
        qg, sl = m["query_garment"], m["slot_label"]
        sim = m["slot_score"]
        qtxt = " ".join(filter(None, [qg.get("color"), qg.get("category")]))
        stxt = f'{sl["color"]} {sl["category"]}'
        rows.append(
            f'<div class="bind {strength_class(sim)}">'
            f'<span class="q">{swatch(qg.get("color","")) if qg.get("color") else ""}{html.escape(qtxt)}</span>'
            f'<span class="arr">→</span>'
            f'<span class="s">{swatch(sl["color"])}{html.escape(stxt)}</span>'
            f'<span class="sim">{sim:.2f}</span></div>')
    return "".join(rows)


def render_frame(r: dict) -> str:
    src = "FP" if r["source"] == "fashionpedia" else "COCO"
    return f'''<figure class="frame">
      <div class="img"><img loading="lazy" src="{thumb(r["image_path"])}" alt="result {r['rank']}">
        <span class="fno">{r['rank']:02d}</span>
        <span class="src {src.lower()}">{src}</span></div>
      <figcaption>
        <div class="meta"><span class="score">{r['score']:.3f}</span>
          <span class="scene">{html.escape(r['scene_label'] or '—')}</span></div>
        {render_matches(r['matches'])}
      </figcaption>
    </figure>'''


def render_query(i: int, q: dict) -> str:
    p = q["parsed"]
    chips = [chip_garment(g) for g in p["garments"]]
    if p["scene"]:
        chips.append(f'<span class="chip ctx">scene · {html.escape(p["scene"])}</span>')
    if p["style_vibe"]:
        chips.append(f'<span class="chip ctx">vibe · {html.escape(p["style_vibe"])}</span>')
    frames = "".join(render_frame(r) for r in q["results"])
    return f'''<section class="query">
      <header class="qhead">
        <span class="qno">Q{i:02d}</span>
        <h2>{html.escape(q["query"])}</h2>
        <div class="parsed">{"".join(chips)}</div>
      </header>
      <div class="sheet">{frames}</div>
    </section>'''


def main() -> None:
    sections = "".join(render_query(i + 1, q) for i, q in enumerate(DATA))
    page = TEMPLATE.replace("{{SECTIONS}}", sections)
    OUT.write_text(page)
    print(f"wrote {OUT}  ({len(page)//1024} KB)")


TEMPLATE = r"""<title>Glance — Retrieval Contact Sheet</title>
<style>
:root{
  --paper:#eeedea; --panel:#f6f5f2; --ink:#191a1d; --muted:#6c6f75; --faint:#9a9ca1;
  --line:#dcdad5; --accent:#2947c9; --hit:#1f8a4c; --part:#b7791f; --miss:#c0392b;
  --shadow:0 1px 2px rgba(20,20,25,.06),0 6px 20px rgba(20,20,25,.06);
}
@media (prefers-color-scheme:dark){:root{
  --paper:#161718; --panel:#1e1f21; --ink:#eceae6; --muted:#9fa1a6; --faint:#6a6c71;
  --line:#2c2e31; --accent:#8aa0ff; --hit:#4bbd7c; --part:#e0a94a; --miss:#e0685c;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.35);
}}
:root[data-theme="light"]{
  --paper:#eeedea; --panel:#f6f5f2; --ink:#191a1d; --muted:#6c6f75; --faint:#9a9ca1;
  --line:#dcdad5; --accent:#2947c9; --hit:#1f8a4c; --part:#b7791f; --miss:#c0392b;
  --shadow:0 1px 2px rgba(20,20,25,.06),0 6px 20px rgba(20,20,25,.06);
}
:root[data-theme="dark"]{
  --paper:#161718; --panel:#1e1f21; --ink:#eceae6; --muted:#9fa1a6; --faint:#6a6c71;
  --line:#2c2e31; --accent:#8aa0ff; --hit:#4bbd7c; --part:#e0a94a; --miss:#e0685c;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(28px,5vw,64px) clamp(18px,4vw,44px)}
.serif{font-family:Georgia,"Iowan Old Style","Times New Roman",serif}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;font-variant-numeric:tabular-nums}

/* ---- masthead ---- */
.mast{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:8px}
.kicker{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-weight:600}
.mast h1{font-family:Georgia,"Iowan Old Style",serif;font-weight:600;
  font-size:clamp(30px,5.2vw,50px);line-height:1.04;margin:.32em 0 .28em;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:62ch;font-size:clamp(15px,1.6vw,17px)}
.lede b{color:var(--ink);font-weight:600}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}
.stat{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;box-shadow:var(--shadow)}
.stat .n{font-size:26px;font-weight:700;letter-spacing:-.02em}
.stat.win .n{color:var(--hit)}
.stat .l{font-size:12px;color:var(--muted);margin-top:2px}
.stat .l b{color:var(--ink)}

/* ---- query section ---- */
.query{padding:38px 0;border-bottom:1px solid var(--line)}
.query:last-child{border-bottom:0}
.qhead{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;align-items:baseline;margin-bottom:22px}
.qno{grid-row:1/3;font-family:Georgia,serif;font-size:clamp(26px,4vw,40px);color:var(--accent);
  font-weight:600;letter-spacing:-.02em;opacity:.9}
.qhead h2{margin:0;font-weight:500;font-size:clamp(19px,2.5vw,26px);font-style:italic;
  font-family:Georgia,"Iowan Old Style",serif;text-wrap:balance;color:var(--ink)}
.parsed{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:4px 11px;font-size:13px;color:var(--ink);text-transform:capitalize}
.chip.ctx{color:var(--muted);text-transform:none;font-size:12px;letter-spacing:.02em}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;flex:none}

/* ---- contact sheet ---- */
.sheet{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
@media (max-width:900px){.sheet{grid-template-columns:repeat(3,1fr)}}
@media (max-width:560px){.sheet{grid-template-columns:repeat(2,1fr)}}
.frame{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;
  box-shadow:var(--shadow);display:flex;flex-direction:column}
.frame .img{position:relative;aspect-ratio:4/5;background:#0000000d;overflow:hidden}
.frame img{width:100%;height:100%;object-fit:cover;display:block}
.fno{position:absolute;left:8px;top:8px;font-family:ui-monospace,Menlo,monospace;font-size:11px;
  font-weight:600;color:#fff;background:rgba(15,16,20,.62);padding:2px 6px;border-radius:6px;
  backdrop-filter:blur(3px)}
.src{position:absolute;right:8px;top:8px;font-size:10px;letter-spacing:.08em;font-weight:700;
  color:#fff;padding:2px 6px;border-radius:6px;background:rgba(15,16,20,.55);backdrop-filter:blur(3px)}
.src.fp{background:rgba(41,71,201,.82)}
figcaption{padding:9px 10px 11px;display:flex;flex-direction:column;gap:6px}
.meta{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
.score{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;font-weight:700;font-size:14px}
.scene{font-size:11px;color:var(--muted);text-transform:capitalize;text-align:right}
.bind{display:flex;align-items:center;gap:5px;font-size:11.5px;line-height:1.3;
  padding:4px 6px;border-radius:7px;background:color-mix(in srgb,var(--ink) 4%,transparent);flex-wrap:wrap}
.bind .q,.bind .s{display:inline-flex;align-items:center;gap:4px;text-transform:capitalize}
.bind .arr{color:var(--faint)}
.bind .sim{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:11px;
  padding:1px 5px;border-radius:5px}
.bind.hit .sim{color:#fff;background:var(--hit)}
.bind.part .sim{color:#fff;background:var(--part)}
.bind.miss .sim{color:#fff;background:var(--miss)}
.bind.none{color:var(--muted);font-style:italic;justify-content:center}
footer{color:var(--faint);font-size:12.5px;margin-top:30px;padding-top:18px;border-top:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center}
footer .legend{display:inline-flex;align-items:center;gap:5px}
footer .dot{width:9px;height:9px;border-radius:3px;display:inline-block}
</style>

<div class="wrap">
  <header class="mast">
    <div class="kicker">Glance · Grounded Axis-Decomposed Retrieval</div>
    <h1>Five queries, and the garments they were bound to.</h1>
    <p class="lede">Each query is parsed into per-garment constraints, then matched against
      image <b>slots</b> whose colour was read from that garment's own pixels. Every frame below
      shows its <b>binding evidence</b> — which requested garment landed on which slot, and how
      strongly. On the compositional swap test this binding scores <b>100%</b> where pooled
      text-image models sit near chance.</p>
    <div class="stats">
      <div class="stat win"><div class="n">100.0%</div><div class="l"><b>G-ADR</b> swap accuracy · 1,259 pairs</div></div>
      <div class="stat"><div class="n">57.5%</div><div class="l">vanilla CLIP (ViT-B/32)</div></div>
      <div class="stat"><div class="n">61.2%</div><div class="l">FashionCLIP</div></div>
      <div class="stat"><div class="n">2,275</div><div class="l">corpus images · 5,190 garment slots</div></div>
    </div>
  </header>

  {{SECTIONS}}

  <footer>
    <span>Top-5 retrieved per query · score = weighted garment binding + scene + vibe.</span>
    <span class="legend"><span class="dot" style="background:var(--hit)"></span>strong ≥ .50</span>
    <span class="legend"><span class="dot" style="background:var(--part)"></span>partial .25–.50</span>
    <span class="legend"><span class="dot" style="background:var(--miss)"></span>miss &lt; .25</span>
    <span class="legend"><span class="dot" style="background:var(--accent)"></span>FP = Fashionpedia · COCO = scene supplement</span>
  </footer>
</div>"""


if __name__ == "__main__":
    main()
