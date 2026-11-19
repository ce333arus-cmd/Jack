"""
Recreation of the cybersigil / neo-tribal motorcycle decals.

Every shape is built as real geometry (shapely), merged into one silhouette,
then exported as:
  * <name>_cut.svg     - cut-ready: the thin white outline as filled paths
                         (what a vinyl plotter / printer needs)
  * <name>_stroke.svg  - the same silhouette as an editable centre-line stroke
  * <name>_solid.svg   - filled silhouette (alt style / weeding reference)
All units are millimetres.

Run:  python3 stickers/generate.py
"""
import math
import os

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "svg")
LINE = 1.6  # outline width in mm (looks like ~1.5 mm on the bike)
WHITE = "#ffffff"


# --------------------------------------------------------------------------- #
# geometry primitives
# --------------------------------------------------------------------------- #
def bez(P, t):
    P = np.asarray(P, float)
    t = np.asarray(t, float)[:, None]
    return ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] \
        + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]


def frame(P, t):
    c = bez(P, t)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    return c, d, n


def width_profile(t, w0, w1, power, swell):
    return w1 + (w0 - w1) * (1 - t) ** power + swell * np.sin(np.pi * t)


def blade(P, w0, w1=0.0, power=1.0, swell=0.0, n=60):
    """Tapered ribbon along a cubic bezier (w0 at start -> w1 at end)."""
    t = np.linspace(0, 1, n)
    c, _, nrm = frame(P, t)
    w = width_profile(t, w0, w1, power, swell) / 2
    L = c + nrm * w[:, None]
    R = c - nrm * w[:, None]
    return Polygon(np.vstack([L, R[::-1]])).buffer(0)


def spike(p, ang, length, width, bend=0.25, power=1.3):
    """Curved thorn from point p, heading `ang` degrees, curling by `bend`."""
    a = math.radians(ang)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    p = np.asarray(p, float)
    P = [p,
         p + d * length * 0.35 + n * bend * length * 0.35,
         p + d * length * 0.70 + n * bend * length * 0.55,
         p + d * length + n * bend * length * 0.35]
    return blade(P, width, 0, power=power)


def thorny(P, w0, rng, thorns=5, tlen=0.22, sides=(1, -1), back=42,
           bend=0.35, inner=True, power=1.05, swell=0.0, start=0.12, stop=0.8):
    """Main tapered blade with thorns along it, thorns pointing toward the tip."""
    parts = [blade(P, w0, 0, power=power, swell=swell)]
    P = np.asarray(P, float)
    total = np.sum(np.linalg.norm(np.diff(bez(P, np.linspace(0, 1, 40)), axis=0), axis=1))
    ts = np.linspace(start, stop, thorns) + rng.uniform(-0.03, 0.03, thorns)
    for i, t in enumerate(ts):
        c, d, n = frame(P, np.array([t - 0.01, t, t + 0.01]))
        c, d, n = c[1], d[1], n[1]
        s = sides[i % len(sides)]
        w = width_profile(np.array([t]), w0, 0, power, swell)[0]
        ang = math.degrees(math.atan2(d[1], d[0])) + s * (back + rng.uniform(-8, 8))
        L = total * tlen * (1 - t * 0.6) * rng.uniform(0.8, 1.15)
        base = c + n * s * w * 0.3
        parts.append(spike(base, ang, L, w * 0.75, bend=-s * bend))
    shape = unary_union(parts)
    if inner and w0 > 9:  # double-line inlay like the original decals
        t = np.linspace(0.05, 0.6, 50)
        c, _, nrm = frame(P, t)
        env = np.sin(np.pi * (t - 0.05) / 0.55) ** 0.7
        w = width_profile(t, w0, 0, power, swell) * 0.16 * env
        cut = Polygon(np.vstack([c + nrm * w[:, None], (c - nrm * w[:, None])[::-1]])).buffer(0)
        shape = shape.difference(cut)
    return shape


def lozenge(p0, p1, width, bulge=0.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    P = [p0, p0 + d / 3 + n * bulge, p0 + 2 * d / 3 + n * bulge, p1]
    return blade(P, 0, 0, swell=width)


def cloud(cx, cy, r, rng, bump=0.38, k=9):
    """Scalloped lobe: a ring of overlapping bubbles."""
    from shapely.geometry import Point
    ang = np.linspace(0, 2 * np.pi, k + 1)[:-1] + rng.uniform(0, 1)
    parts = [Point(cx, cy).buffer(r * 0.75, 48)]
    for a in ang:
        parts.append(Point(cx + r * 0.75 * math.cos(a), cy + r * 0.75 * math.sin(a))
                     .buffer(r * bump * rng.uniform(0.85, 1.15), 32))
    return unary_union(parts)


def mirror(shape):
    return unary_union([shape, affinity.scale(shape, -1, 1, origin=(0, 0))])


def finish(shape, close=1.2):
    """Merge near-touching parts while keeping thorn tips sharp."""
    s = shape.buffer(close, join_style=1).buffer(-close, join_style=2, mitre_limit=8)
    s = unary_union([s, shape])
    return s.simplify(0.04)


# --------------------------------------------------------------------------- #
# svg output
# --------------------------------------------------------------------------- #
def ring_d(coords):
    pts = list(coords)
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts[:-1]) + " Z"


def geom_d(g):
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))
    out = []
    for p in polys:
        if p.is_empty or not isinstance(p, Polygon):
            continue
        out.append(ring_d(p.exterior.coords))
        out.extend(ring_d(r.coords) for r in p.interiors)
    return " ".join(out)


def svg_doc(body, bounds, pad=6, title=""):
    x0, y0, x1, y1 = bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    w, h = x1 - x0, y1 - y0
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" height="{h:.1f}mm" '
            f'viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}">\n'
            f'<title>{title}</title>\n{body}\n</svg>\n')


def export(name, shape, title):
    shape = finish(shape)
    b = shape.bounds
    # cut-ready outline: ring of constant width straddling the silhouette edge
    ring = shape.buffer(LINE / 2, join_style=1).difference(
        shape.buffer(-LINE / 2, join_style=2, mitre_limit=8))
    ring = ring.simplify(0.03)
    files = {
        "cut": f'<path fill="{WHITE}" fill-rule="evenodd" d="{geom_d(ring)}"/>',
        "stroke": f'<path fill="none" stroke="{WHITE}" stroke-width="{LINE}" '
                  f'stroke-linejoin="miter" stroke-miterlimit="8" d="{geom_d(shape)}"/>',
        "solid": f'<path fill="{WHITE}" fill-rule="evenodd" d="{geom_d(shape)}"/>',
    }
    for kind, body in files.items():
        with open(os.path.join(OUT, f"{name}_{kind}.svg"), "w") as f:
            f.write(svg_doc(body, b, title=f"{title} ({kind})"))
    print(f"{name}: {b[2]-b[0]:.0f} x {b[3]-b[1]:.0f} mm")


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #
def side_fairing(rng):
    """Big horizontal piece on the mid fairing (both sides, mirror for the left)."""
    half = []
    # central spine / vertebra
    half.append(lozenge((0, -100), (0, 100), 30))
    for y in (-55, -20, 15, 50):
        half.append(lozenge((-2, y), (26, y + 6), 9, bulge=-4))
    # ribs / claws curling down and out
    for i, (y, reach, drop) in enumerate([(-60, 150, 70), (-28, 190, 85),
                                          (6, 165, 80), (38, 120, 60)]):
        P = [(8, y), (8 + reach * 0.45, y - 45 + i * 8),
             (8 + reach * 0.95, y - 10), (8 + reach, y + drop)]
        half.append(thorny(P, 22 - i * 2, rng, thorns=5, tlen=0.18, sides=(-1, -1, 1), back=48))
        # hooked claw tip curling back in
        end = np.array(P[3], float)
        half.append(spike(end - (0, 8), 150, 40, 8, bend=-0.7))
    # big swept horn on top
    half.append(thorny([(10, -70), (60, -140), (170, -150), (285, -105)], 30, rng,
                       thorns=7, tlen=0.15, sides=(-1, -1, 1), back=40))
    half.append(thorny([(14, -50), (80, -95), (150, -90), (215, -60)], 18, rng,
                       thorns=4, tlen=0.16, sides=(1,), back=45))
    # long tail tendril
    half.append(thorny([(15, 70), (90, 140), (210, 40), (330, 70)], 22, rng,
                       thorns=8, tlen=0.14, sides=(1, -1), back=45))
    half.append(thorny([(20, 85), (70, 120), (130, 110), (175, 140)], 13, rng,
                       thorns=3, tlen=0.2, sides=(1,), back=45))
    # flame tips above the spine
    half.append(spike((4, -85), -95, 70, 14, bend=-0.4))
    half.append(spike((10, -70), -60, 55, 10, bend=0.5))
    # small barbs at the base
    half.append(spike((6, 88), 95, 50, 12, bend=0.3))
    half.append(spike((20, 70), 60, 45, 9, bend=-0.4))
    return mirror(unary_union(half))


def lower_fairing(rng):
    """Long low piece under the main one: claws reaching forward."""
    parts = []
    parts.append(thorny([(-260, 10), (-150, -50), (0, 40), (230, -20)], 18, rng,
                        thorns=9, tlen=0.06, sides=(1, -1), back=45))
    for x, s in ((-170, 1), (-60, -1), (60, 1), (150, -1)):
        P = [(x, 0), (x + 40, -60 * s), (x + 110, -55 * s), (x + 130, -10 * s)]
        parts.append(thorny(P, 12, rng, thorns=3, tlen=0.2, sides=(s,), back=45))
    parts.append(thorny([(200, -10), (260, -60), (300, 20), (250, 50)], 12, rng,
                        thorns=3, tlen=0.25, sides=(1,)))
    parts.append(spike((-250, 8), 190, 80, 14, bend=0.3))
    return unary_union(parts)


def front_cowl(rng):
    """Tall piece on the nose: vertical thorn trunk ending in lobed curls."""
    half = []
    half.append(lozenge((0, -230), (0, 200), 30))
    # branches climbing up the trunk
    for i, y in enumerate((-170, -110, -50, 10)):
        P = [(4, y), (40, y - 30), (80, y - 10), (95 - i * 6, y - 70 + i * 4)]
        half.append(thorny(P, 20 - i * 1.5, rng, thorns=4, tlen=0.25, sides=(1, -1), back=45))
    # lobed curls at the bottom
    for cx, cy, r in ((30, 90, 38), (75, 140, 34), (25, 170, 30), (90, 70, 26)):
        half.append(cloud(cx, cy, r, rng))
        for a in rng.uniform(0, 2 * np.pi, 3):
            p = (cx + r * 0.8 * math.cos(a), cy + r * 0.8 * math.sin(a))
            half.append(spike(p, math.degrees(a) + 25, r * 0.7, r * 0.22, bend=-0.6))
    half.append(spike((2, -225), -90, 60, 14, bend=0.2))
    shape = mirror(unary_union(half))
    # eye slits inside the lobes for the inner-outline look
    for cx, cy, r in ((30, 90, 38), (75, 140, 34), (25, 170, 30), (90, 70, 26)):
        for sx in (1, -1):
            inner = cloud(sx * cx, cy, r * 0.62, np.random.default_rng(int(cx + cy)), bump=0.3)
            shape = shape.difference(inner.difference(
                lozenge((sx * (cx - r * 0.5), cy + r * 0.1), (sx * (cx + r * 0.5), cy - r * 0.2), r * 0.18)))
    return shape


def tail_piece(rng):
    """Small piece for the tail / seat cowl and tank."""
    half = []
    half.append(lozenge((0, -60), (0, 60), 14))
    half.append(thorny([(4, -30), (50, -80), (110, -60), (150, -10)], 11, rng,
                       thorns=4, tlen=0.2, sides=(-1,)))
    half.append(thorny([(4, 20), (50, 70), (100, 40), (130, 70)], 9, rng,
                       thorns=3, tlen=0.22, sides=(1,)))
    half.append(spike((2, -55), -90, 40, 10, bend=0.3))
    return mirror(unary_union(half))


# --------------------------------------------------------------------------- #
# lettering
# --------------------------------------------------------------------------- #
def text_svg(name, lines, font_path, size_mm, leading=1.05, align="center",
             xscale=1.0, title=""):
    font = TTFont(font_path)
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    s = size_mm / upm
    asc = font["hhea"].ascent
    widths = []
    for line in lines:
        widths.append(sum(font["hmtx"][cmap[ord(ch)]][0] for ch in line) * s * xscale)
    maxw = max(widths)
    paths, y = [], 0.0
    for line, lw in zip(lines, widths):
        x = {"center": (maxw - lw) / 2, "right": maxw - lw, "left": 0}[align]
        y += asc * s if not paths else size_mm * leading
        for ch in line:
            g = cmap[ord(ch)]
            pen = SVGPathPen(gs)
            gs[g].draw(TransformPen(pen, (s * xscale, 0, 0, -s, x, y)))
            d = pen.getCommands()
            if d:
                paths.append(d)
            x += font["hmtx"][g][0] * s * xscale
    h = y + size_mm * 0.3
    body = f'<path fill="{WHITE}" fill-rule="nonzero" d="{" ".join(paths)}"/>'
    with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
        f.write(svg_doc(body, (0, 0, maxw, h), pad=3, title=title))
    print(f"{name}: {maxw:.0f} x {h:.0f} mm")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    export("01_side_fairing", side_fairing(rng), "Side fairing sigil")
    export("02_lower_fairing", lower_fairing(rng), "Lower fairing claws")
    export("03_front_cowl", front_cowl(rng), "Front cowl sigil")
    export("04_tail_tank", tail_piece(rng), "Tail / tank sigil")
    text_svg("05_windscreen_text", ["GOTH GF", "RAPID", "RESPONSE", "UNIT"],
             os.path.join(HERE, "fonts", "UnifrakturMaguntia.woff2"), 22,
             leading=1.0, title="GOTH GF RAPID RESPONSE UNIT")
    inter = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
    if os.path.exists(inter):
        text_svg("06_no_step", ["NO STEP"], inter, 14, xscale=0.82, title="NO STEP")


if __name__ == "__main__":
    main()
