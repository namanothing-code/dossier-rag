"""Identify undecodable PDF glyphs by their shape.

Some PDFs embed fonts without a usable character map, so text extraction yields
(cid:N), a control character, or nothing for symbols such as ± ≤ ≥ ° µ •.
The glyph's shape is still in the PDF: we render it, then compare it with
reference renderings of candidate symbols in several fonts. A match is accepted
only when it is clearly better than the runner-up; otherwise the (cid:N)
marker stays so the symbol is checked by hand.

All occurrences of the same (font, cid) are identified together, using the
average of their scores, which makes the decision more reliable.
"""
import glob
from functools import lru_cache

import numpy as np

CANDIDATES = ("±×÷≤≥≈≠°µ–—•·‘’“”†‡§¶©®™→←↑↓⇒½¼¾²³‰√∞Δαβγδελπσ"
              "◦▪■□●○✓…′″<>+=−%*/~^")
GRID = 32
SCALE = 8          # render scale (1 pt -> 8 px, ~576 dpi)
ACCEPT = 0.72      # minimum combined score
MARGIN = 0.04      # required lead over the best symbol with a different meaning
# shapes that differ only in style; a close race between them is not ambiguous
# (ordered: the first character is the one written to the text)
EQUIV = ["•●", "◦○", "▪■", "’‘′", "”“″"]
# shapes that are easily confused at small sizes need a clearer lead
HARD = {"¼": 0.12, "½": 0.12, "¾": 0.12, "λ": 0.10, "π": 0.10, "γ": 0.10}


def _same(a, b):
    return a == b or any(a in g and b in g for g in EQUIV)


CANON = ["•●", "◦○", "▪■"]  # style variants written as one character


def _canonical(ch):
    for g in CANON:
        if ch in g:
            return g[0]
    return ch

FONT_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation*/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation*/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/seguisym.ttf",
]


def _fonts():
    out = []
    for g in FONT_GLOBS:
        out += sorted(glob.glob(g))[:1]
    if not out:
        try:
            import matplotlib
            out += glob.glob(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSerif.ttf")
        except Exception:
            pass
    return out


def _features(ink, box_h, offset=0):
    """ink: bool array of the glyph's font box (rows = top..bottom). Returns
    (normalised GRIDxGRID bitmap, aspect, relative height, relative vertical centre)."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = ink[y0:y1, x0:x1].astype(np.float32)
    h, w = g.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.float32)
    pad[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
    from PIL import Image, ImageFilter
    im = Image.fromarray((pad * 255).astype(np.uint8)).resize((GRID, GRID), Image.BOX)
    sharp = np.asarray(im, np.float32) / 255.0
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
    # stroke profiles: how much ink each row / column holds (counts bars and stems)
    rows = (sharp > 0.35).mean(axis=1)
    cols = (sharp > 0.35).mean(axis=0)
    return (blur, rows, cols), w / h, h / box_h, ((y0 + y1) / 2 - offset) / box_h


@lru_cache(maxsize=None)
def _references():
    """[(char, features)] for every candidate in every available font."""
    from PIL import Image, ImageDraw, ImageFont
    refs = []
    size = 96
    for path in _fonts():
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            continue
        for ch in CANDIDATES:
            # skip characters the font lacks (they render as .notdef boxes)
            try:
                if font.getmask(ch).getbbox() is None:
                    continue
            except Exception:
                continue
            img = Image.new("L", (size * 3, size * 3), 0)
            d = ImageDraw.Draw(img)
            base = int(size * 1.6)
            d.text((size, base), ch, font=font, fill=255, anchor="ls")
            # em box like pdfminer: baseline + 0.2*size down, size tall; widened like the PDF crop
            top, bottom = base - int(size * 0.8), base + int(size * 0.2)
            pad = int(size * 0.35)
            ink = np.asarray(img)[top - pad:bottom + pad, :] > 110
            f = _features(ink, bottom - top, offset=pad)
            if f:
                refs.append((ch, f))
    return refs


def _profile_sim(p, q):
    """Similarity of two ink profiles, tolerant to a 1-cell shift."""
    best = 0.0
    for sh in (-1, 0, 1):
        r = np.roll(q, sh)
        best = max(best, 1 - np.abs(p - r).sum() / (p.sum() + r.sum() + 1e-6))
    return best


def _bars(prof):
    """Number of strong horizontal (or vertical) bars: runs where the profile is high."""
    hi = prof > 0.55
    return int(np.sum(hi[1:] & ~hi[:-1]) + hi[0])


def _score(a, b):
    (bmp_a, rows_a, cols_a), asp_a, hh_a, cy_a = a
    (bmp_b, rows_b, cols_b), asp_b, hh_b, cy_b = b
    inter = np.minimum(bmp_a, bmp_b).sum()
    dice = 2 * inter / (bmp_a.sum() + bmp_b.sum() + 1e-6)
    prof = 0.5 * _profile_sim(rows_a, rows_b) + 0.5 * _profile_sim(cols_a, cols_b)
    bars = 1.0 if (_bars(rows_a), _bars(cols_a)) == (_bars(rows_b), _bars(cols_b)) else 0.0
    asp = 1 - min(1.0, abs(np.log((asp_a + 1e-3) / (asp_b + 1e-3))) / 1.0)
    hgt = 1 - min(1.0, abs(hh_a - hh_b) / 0.3)
    pos = 1 - min(1.0, abs(cy_a - cy_b) / 0.4)
    return 0.40 * dice + 0.15 * prof + 0.15 * bars + 0.12 * asp + 0.13 * hgt + 0.05 * pos


def _components_inside(ink):
    """Drop connected ink components that touch the top or bottom edge of the widened
    crop: those are strokes of the text lines above and below. Separate parts of the
    glyph itself (the lower bar of ±, the dots of ÷) are kept."""
    h, w = ink.shape
    lab = np.zeros((h, w), np.int32)
    keep = np.zeros((h, w), bool)
    n = 0
    for y in range(h):
        for x in range(w):
            if ink[y, x] and not lab[y, x]:
                n += 1
                stack, pix = [(y, x)], []
                lab[y, x] = n
                while stack:
                    a, b = stack.pop()
                    pix.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            u, v = a + da, b + db
                            if 0 <= u < h and 0 <= v < w and ink[u, v] and not lab[u, v]:
                                lab[u, v] = n
                                stack.append((u, v))
                rows = [q[0] for q in pix]
                if min(rows) > 0 and max(rows) < h - 1:
                    for a, b in pix:
                        keep[a, b] = True
    return keep


def glyph_features(pdf_doc, page_cache, pn, c):
    """Render one pdfplumber char (dict) and return its features."""
    if pn not in page_cache:
        page_cache[pn] = np.asarray(pdf_doc[pn - 1].render(scale=SCALE, grayscale=True).to_pil())
    img = page_cache[pn]
    size = c.get("size") or (c["bottom"] - c["top"])
    # The reported box is unreliable for symbol fonts (glyphs can hang below it),
    # so crop a band 35% of the size larger on each side, then keep only the ink
    # connected to the reported box.
    bottom = c["bottom"]
    top = bottom - size
    pad = 0.35 * size
    x0, x1 = c["x0"] + 0.15, c["x1"] - 0.15
    ya, yb = int((top - pad) * SCALE), int((bottom + pad) * SCALE)
    xa, xb = int(x0 * SCALE), int(x1 * SCALE)
    ya = max(0, ya)
    crop = img[ya:yb, max(0, xa):xb] < 128
    if crop.size == 0 or not crop.any():
        return None
    ink = _components_inside(crop)
    # features relative to the reported box (height = font size, starting at `top`)
    return _features(ink, int(size * SCALE), offset=int(top * SCALE) - ya)


def rank(features_list):
    """Average scores over all samples of one glyph. Returns [(char, score)] best first."""
    refs = _references()
    best = {}
    for f in features_list:
        per = {}
        for ch, rf in refs:
            s = _score(f, rf)
            if s > per.get(ch, -1):
                per[ch] = s
        for ch, s in per.items():
            best[ch] = best.get(ch, 0) + s / len(features_list)
    return sorted(best.items(), key=lambda kv: -kv[1])


def identify(path, glyph_chars):
    """glyph_chars: {(fontname, cid_text): [(page, char_dict), ...]}.
    Returns {(fontname, cid_text): (symbol or None, score, runner_up)}."""
    if not _references():
        return {}
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    cache, out = {}, {}
    try:
        for key, samples in glyph_chars.items():
            feats = [f for f in (glyph_features(doc, cache, pn, c) for pn, c in samples[:8]) if f]
            if not feats:
                continue
            ranked = rank(feats)
            if not ranked:
                continue
            ch, s = ranked[0]
            ch2, s2 = next(((c2, v) for c2, v in ranked[1:] if not _same(c2, ch)), (None, 0))
            ok = s >= ACCEPT and (s - s2) >= max(MARGIN, HARD.get(ch, 0), HARD.get(ch2, 0))
            out[key] = (_canonical(ch) if ok else None, round(s, 3), (ch2, round(s2, 3)))
    finally:
        doc.close()
    return out
