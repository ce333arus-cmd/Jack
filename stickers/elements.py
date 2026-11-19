"""
Individual thin cybersigil sticker elements.

Each element is ONE solid connected shape (a single cut line for a vinyl plotter):
a thin flowing stem with needle tips and curved thorns blended into it.
Combine several on a panel to build the look - like the reference bike.

All units mm.  Run: python3 elements.py
Output: print/elements/*.svg/.pdf  +  preview/elements_*.png
"""
import math
import os

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "print", "elements")
PRE = os.path.join(HERE, "preview")


# --------------------------------------------------------------------------- #
# smooth paths
# --------------------------------------------------------------------------- #
def spline(pts, n=400):
    """Centripetal Catmull-Rom through control points -> smooth dense polyline."""
    P = np.asarray(pts, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** 0.5
        t0 = 0; t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
        for t in np.linspace(t1, t2, n // (len(P) - 3), endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-2])
    return np.array(out)


def arclen(c):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]


def even(c, step=0.5):
    d = arclen(c)
    s = np.linspace(0, d[-1], max(3, int(d[-1] / step)))
    return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], 1)


def ribbon(c, widths):
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    w = np.asarray(widths)[:, None] / 2
    return Polygon(np.vstack([c + n * w, (c - n * w)[::-1]])).buffer(0)


# width profiles (t in 0..1)  -> fraction of max width
def needle_both(t, a=0.18, b=0.30):
    """Full in the middle, long sharp taper at both ends."""
    up = np.clip(t / a, 0, 1)
    dn = np.clip((1 - t) / b, 0, 1)
    return np.minimum(np.sin(up * np.pi / 2), np.sin(dn * np.pi / 2)) ** 1.1


def crescent(t, p=1.2):
    return np.sin(np.pi * t) ** p


def comet(t, rise=0.08):
    up = np.sin(np.clip(t / rise, 0, 1) * np.pi / 2)
    return up * (1 - t) ** 1.15


class Stem:
    def __init__(self, pts, W, profile=needle_both):
        self.c = even(spline(pts))
        self.t = arclen(self.c) / arclen(self.c)[-1]
        self.w = W * profile(self.t)
        self.poly = ribbon(self.c, self.w)

    def at(self, t):
        i = int(np.clip(t, 0, 1) * (len(self.c) - 1))
        i = min(max(i, 1), len(self.c) - 2)
        d = self.c[i + 1] - self.c[i - 1]
        d /= np.linalg.norm(d)
        return self.c[i], d, np.array([-d[1], d[0]]), self.w[i]


def thorn(stem, t, side, length, angle=38, curl=0.35, base=None):
    """Curved thorn growing out of the stem at t, leaning toward the stem's tip."""
    p, d, n, w = stem.at(t)
    base_w = base if base is not None else max(w * 1.7, 3.0)
    length *= 1.25
    a = math.atan2(d[1], d[0]) + side * math.radians(angle)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    start = p + n * side * w * 0.15 - d * base_w * 0.2   # root buried in the stem, own side only
    pts = [start,
           start + u * length * 0.35,
           start + u * length * 0.7 - v * side * curl * length * 0.18,
           start + u * length - v * side * curl * length * 0.45]
    c = even(spline(pts, 200), 0.3)
    tt = arclen(c) / arclen(c)[-1]
    th = ribbon(c, base_w * (1 - tt) ** 1.35 + 0.01)
    # trim anything that pokes out on the far side of the stem
    far = ribbon(stem.c, stem.w + 40).difference(ribbon(stem.c, stem.w))
    cut = [g for g in getattr(far, "geoms", [far])
           if g.distance(Point(*(p - n * side * (w / 2 + 3)))) < 1e-6]
    if cut:
        th = th.difference(unary_union(cut))
    return th


def finish(parts, fillet=0.9):
    """Union + blended (filleted) joints. Concave corners get rounded, tips stay sharp."""
    g = unary_union(parts)
    g = g.buffer(fillet, join_style=1).buffer(-fillet, join_style=2, mitre_limit=10)
    g = unary_union([g, unary_union(parts)])
    if g.geom_type == "MultiPolygon":            # keep it one sticker
        g = max(g.geoms, key=lambda q: q.area)
    return g.simplify(0.02)


# --------------------------------------------------------------------------- #
# the five elements
# --------------------------------------------------------------------------- #
def el_claw():
    """Long sweeping crescent claw with one barb near its root."""
    s = Stem([(0, 0), (70, -40), (170, -55), (270, -25), (330, 30)], 7.5,
             profile=lambda t: crescent(t, 1.0) * (1 - 0.35 * t))
    return finish([s.poly, thorn(s, 0.22, -1, 26, angle=40, curl=0.5)])


def el_vine():
    """Flowing S-shaped thorn vine, thorns alternating, shrinking toward the tips."""
    s = Stem([(0, 40), (80, 10), (170, 30), (260, 70), (350, 50), (420, 0)], 3.4)
    parts = [s.poly]
    for i, t in enumerate(np.linspace(0.14, 0.86, 8)):
        side = 1 if i % 2 else -1
        L = 30 * (1 - abs(t - 0.45) * 1.1)
        parts.append(thorn(s, t, side, max(L, 12), angle=36 + 6 * (i % 3), curl=0.45))
    return finish(parts)


def el_hook():
    """Sweep that curls back into a sharp hook, thorns on the outside of the curve."""
    s = Stem([(0, 0), (90, -30), (190, -30), (260, 10), (275, 70), (240, 105), (200, 90)], 4.2,
             profile=lambda t: needle_both(t, 0.15, 0.22))
    parts = [s.poly]
    for t, L in ((0.25, 26), (0.42, 30), (0.58, 24), (0.72, 16)):
        parts.append(thorn(s, t, -1, L, angle=40, curl=0.5))
    parts.append(thorn(s, 0.35, 1, 14, angle=45, curl=0.4))
    return finish(parts)


def el_flame():
    """Three blades fanning out of one root, like a tribal flame."""
    root = (0, 0)
    a = Stem([root, (60, -25), (150, -70), (250, -85)], 6.0, profile=comet)
    b = Stem([root, (70, -5), (150, -20), (215, -10)], 5.0, profile=comet)
    c = Stem([root, (60, 15), (120, 30), (165, 55)], 4.2, profile=comet)
    parts = [a.poly, b.poly, c.poly,
             thorn(a, 0.45, -1, 22, angle=40, curl=0.5),
             thorn(b, 0.5, -1, 16, angle=42, curl=0.5),
             thorn(c, 0.45, 1, 14, angle=40, curl=0.5)]
    # tail spike behind the root
    parts.append(Stem([(6, 0), (-30, 8), (-70, 25)], 5.0, profile=comet).poly)
    return finish(parts)


def el_spine():
    """Symmetrical vertical sigil: thin spine, mirrored curved thorns, forked top."""
    s = Stem([(0, 260), (0, 180), (0, 90), (0, 0)], 5.0, profile=lambda t: needle_both(t, 0.12, 0.25))
    parts = [s.poly]
    for t, L in ((0.28, 46), (0.45, 40), (0.6, 32), (0.73, 22)):
        for side in (1, -1):
            parts.append(thorn(s, t, side, L, angle=58, curl=0.6))
    for side in (1, -1):  # forked horns near the top
        parts.append(Stem([(0, 40), (side * 14, 15), (side * 18, -15), (side * 10, -45)], 3.6,
                          profile=comet).poly)
    return finish(parts)


ELEMENTS = {
    "el1_claw": (el_claw, "Crescent claw"),
    "el2_vine": (el_vine, "Thorn vine"),
    "el3_hook": (el_hook, "Thorn hook"),
    "el4_flame": (el_flame, "Flame blades"),
    "el5_spine": (el_spine, "Sigil spine"),
}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def geom_d(g):
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    out = []
    for q in polys:
        for r in [q.exterior] + list(q.interiors):
            pts = list(r.coords)[:-1]
            out.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z")
    return " ".join(out)


def svg(g, color="#ffffff", bg=None, pad=4, title=""):
    x0, y0, x1, y1 = g.bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    w, h = x1 - x0, y1 - y0
    r = f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{bg}"/>' if bg else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w:.1f}mm" height="{h:.1f}mm" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}">'
            f'<title>{title}</title>{r}<path fill="{color}" fill-rule="evenodd" d="{geom_d(g)}"/></svg>\n'), w, h


def save(name, g, title, png_w=1600):
    import cairosvg
    os.makedirs(OUT, exist_ok=True)
    s, w, h = svg(g, title=title)
    open(os.path.join(OUT, name + ".svg"), "w").write(s)
    open(os.path.join(OUT, name + "_black.svg"), "w").write(svg(g, "#000000", title=title)[0])
    cairosvg.svg2pdf(bytestring=s.encode(), write_to=os.path.join(OUT, name + ".pdf"))
    cairosvg.svg2png(bytestring=svg(g, bg="#111111", pad=10)[0].encode(),
                     write_to=os.path.join(PRE, f"elements_{name}.png"), output_width=png_w)
    return w, h


def place(g, x, y, rot=0, scale=1, flip=False):
    if flip:
        g = affinity.scale(g, -1, 1, origin="centroid")
    g = affinity.rotate(g, rot, origin="centroid")
    g = affinity.scale(g, scale, scale, origin="centroid")
    c = g.centroid
    return affinity.translate(g, x - c.x, y - c.y)


def composition(els, gap=7.0):
    """Example panel: elements nested so their curves run alongside each other, no crossings.
    Each element is dropped in from above until it sits `gap` mm under the ones already placed."""
    E = els
    plan = [  # (element, x centre, rotation, scale, flip)
        ("el1_claw", 330, -4, 1.0, True),
        ("el2_vine", 330, -2, 1.0, True),
        ("el4_flame", 380, 4, 1.0, True),
        ("el1_claw", 300, 2, 0.85, True),
        ("el3_hook", 150, 0, 0.8, True),
        ("el2_vine", 400, 6, 0.7, True),
    ]
    placed = []
    for name, x, rot, sc, flip in plan:
        y = -200.0
        g = place(E[name], x, y, rot, sc, flip)
        if placed:
            acc = unary_union(placed)
            # slide down until touching, then back off to the gap
            while g.distance(acc) > gap and g.bounds[1] < acc.bounds[3] + 40:
                g = affinity.translate(g, 0, 1.0)
            while g.intersects(acc.buffer(gap)):
                g = affinity.translate(g, 0, 1.0)
        placed.append(g)
    return unary_union(placed)


if __name__ == "__main__":
    els = {}
    for name, (fn, title) in ELEMENTS.items():
        g = fn()
        g = affinity.translate(g, -g.bounds[0], -g.bounds[1])
        els[name] = g
        w, h = save(name, g, title)
        print(f"{name}: {w:.0f} x {h:.0f} mm, {g.geom_type}, min stem ~ ok")
    comp = composition(els)
    w, h = save("combo_example", comp, "Example layout", png_w=2000)
    print(f"combo_example: {w:.0f} x {h:.0f} mm")
