"""
Original cybersigil decal set, drawn in the style of the reference bike.

Style recipe (taken from the reference photos):
  1. flowing silhouettes - "comet" blades, heavy at one end, needle-sharp at both,
     drawn as a thin OUTLINE
  2. dense jagged lightning lines inside them, following the flow, with small barbs
  3. long thin outlined crescent claws hanging / radiating off the mass
  4. scalloped thorny lobes (cowl piece)

All geometry in mm. Output: print/sigil_*.svg (+ _black, _mirrored, .pdf)
Run: python3 sigil.py
"""
import math
import os

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "print")
LINE = 1.4      # outline weight (mm)
FILL_LINE = 1.3  # weight of the inner lightning lines (mm)


# --------------------------------------------------------------------------- #
# curve helpers
# --------------------------------------------------------------------------- #
def bez(P, t):
    P = np.asarray(P, float)
    t = np.asarray(t, float)[:, None]
    return ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] \
        + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]


def frame(P, n=120):
    t = np.linspace(0, 1, n)
    c = bez(P, t)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
    return t, c, d, np.stack([-d[:, 1], d[:, 0]], 1)


def comet(t, rise=0.22, tail=1.25):
    """Width profile 0..1: quick rounded rise to full, long taper to a needle."""
    up = np.sin(np.clip(t / rise, 0, 1) * np.pi / 2) ** 0.8
    down = np.clip((1 - t) / (1 - rise), 0, 1) ** tail
    return up * np.where(t < rise, 1, down)


def crescent(t):
    return np.sin(np.pi * t) ** 0.9


class Blade:
    """A filled shape along a cubic bezier with a width profile."""

    def __init__(self, P, W, profile=comet, fill=True, **kw):
        self.P = np.asarray(P, float)
        self.W = W
        self.prof = lambda t: profile(t, **kw)
        self.fill = fill
        t, c, d, n = frame(self.P)
        w = self.W * self.prof(t) / 2
        self.poly = Polygon(np.vstack([c + n * w[:, None], (c - n * w[:, None])[::-1]])).buffer(0)

    def offset_curve(self, f, t0=0.0, t1=1.0, n=200):
        t = np.linspace(t0, t1, n)
        c = bez(self.P, t)
        d = np.gradient(c, axis=0)
        d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        w = self.W * self.prof(t) / 2
        return c + nrm * (f * w)[:, None]


def spike(p, ang, length, width, bend=0.25):
    a = math.radians(ang)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    p = np.asarray(p, float)
    P = [p, p + d * length * 0.35 + n * bend * length * 0.35,
         p + d * length * 0.7 + n * bend * length * 0.5, p + d * length + n * bend * length * 0.3]
    return Blade(P, width, profile=lambda t: (1 - t) ** 1.2, fill=False)


# --------------------------------------------------------------------------- #
# strokes
# --------------------------------------------------------------------------- #
def tapered(p, w0, w1):
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(p))
    w = (w0 + (w1 - w0) * t ** 1.4)[:, None] / 2
    return Polygon(np.vstack([p + nrm * w, (p - nrm * w)[::-1]])).buffer(0)


def resample(p, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < step:
        return p
    s = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


def stroke(p, W, taper=6.0):
    """Constant-width polyline with needle tapers at both ends, mitred corners."""
    p = resample(p, 0.4)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = d[-1]
    if L < 1.0:
        return Polygon()
    tl = min(taper, L * 0.4)
    parts = []
    mid = p[(d >= tl - 0.3) & (d <= L - tl + 0.3)]
    if len(mid) >= 2:
        parts.append(LineString(mid).buffer(W / 2, cap_style=2, join_style=2, mitre_limit=4))
    a = p[d <= tl + 0.4][::-1]
    b = p[d >= L - tl - 0.4]
    if len(a) >= 2:
        parts.append(tapered(a, W, 0))
    if len(b) >= 2:
        parts.append(tapered(b, W, 0))
    return unary_union(parts)


def lightning(p, rng, amp=2.0, step=(4.0, 9.0)):
    """Turn a smooth polyline into a jagged one (sharp alternating kinks)."""
    p = resample(p, 0.5)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = d[-1]
    s, out, sign = 0.0, [], 1
    while s < L:
        i = min(np.searchsorted(d, s), len(p) - 1)
        j = min(i + 1, len(p) - 1)
        t = p[j] - p[max(i - 1, 0)]
        t /= np.linalg.norm(t) + 1e-9
        n = np.array([-t[1], t[0]])
        k = 0 if (s == 0) else sign * amp * rng.uniform(0.3, 1.0)
        out.append(p[i] + n * k)
        sign = -sign
        s += rng.uniform(*step)
    out.append(p[-1])
    return np.array(out)


def barbs(p, rng, W, every=0.3, size=(3.5, 7.5)):
    """Small thorns off the corners of a jagged line, pointing along its direction."""
    res = []
    for i in range(1, len(p) - 1):
        if rng.random() > every:
            continue
        t = p[i + 1] - p[i - 1]
        t /= np.linalg.norm(t) + 1e-9
        s = rng.choice([-1, 1])
        ang = math.degrees(math.atan2(t[1], t[0])) + s * rng.uniform(35, 60)
        res.append(spike(p[i], ang, rng.uniform(*size), W * 1.5, bend=-s * 0.3).poly)
    return res


def fill_lines(bl, rng, spacing=8.5, inset=3.2, seg=(18, 55), gap=(3, 9)):
    """Dense jagged flow lines inside a blade, clipped away from its outline."""
    if bl.W < spacing * 2.2:
        return []
    clip = bl.poly.buffer(-(inset + LINE / 2))
    if clip.is_empty:
        return []
    m = int(bl.W / spacing)
    parts = []
    for f in np.linspace(-0.8, 0.8, m) + rng.uniform(-0.06, 0.06, m):
        curve = bl.offset_curve(f)
        dd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        s = rng.uniform(0, 6)
        while s < dd[-1]:
            e = s + rng.uniform(*seg)
            piece = curve[(dd >= s) & (dd <= e)]
            if len(piece) >= 3:
                jag = lightning(piece, rng)
                g = stroke(jag, FILL_LINE, taper=6)
                parts.extend([g] + barbs(jag, rng, FILL_LINE))
            s = e + rng.uniform(*gap)
    return [unary_union(parts).intersection(clip)] if parts else []


def cells(bl, rng, spacing=10.5, inset=0.5, seg=(22, 60), gap=(3, 8), cw=(4.0, 7.5)):
    """Pack the inside of a blade with small thorn cells that follow its flow.
    Each cell is a slim pointed shape with a kink or two and the odd barb;
    the cells get outlined later, which gives the double-line sigil texture."""
    spacing = float(np.clip(bl.W / 3.0, 7.0, spacing))
    if bl.W < 20:
        return []
    clip = bl.poly.buffer(-(inset + LINE))
    if clip.is_empty:
        return []
    m = max(1, int(bl.W / spacing))
    out = []
    for f in np.linspace(-0.9, 0.9, m) + rng.uniform(-0.05, 0.05, m):
        curve = bl.offset_curve(f, n=300)
        dd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        s = rng.uniform(0, 8)
        while s < dd[-1]:
            e = s + rng.uniform(*seg)
            piece = curve[(dd >= s) & (dd <= e)]
            s = e + rng.uniform(*gap)
            if len(piece) < 4:
                continue
            spine = lightning(piece, rng, amp=rng.uniform(0.6, 1.8), step=(6, 13))
            spine = resample(spine, 0.6)
            t = np.linspace(0, 1, len(spine))
            d = np.gradient(spine, axis=0)
            d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9
            nrm = np.stack([-d[:, 1], d[:, 0]], 1)
            prof = np.sin(np.pi * t) ** 0.85 if rng.random() < 0.5 else comet(t, rise=0.3)
            w = rng.uniform(*cw) * prof / 2
            if len(spine) < 4:
                continue
            cell = Polygon(np.vstack([spine + nrm * w[:, None], (spine - nrm * w[:, None])[::-1]])).buffer(0)
            # barbs off the cell
            for _ in range(rng.integers(0, 3)):
                i = int(rng.uniform(0.25, 0.75) * len(spine))
                sd = rng.choice([-1, 1])
                ang = math.degrees(math.atan2(d[i][1], d[i][0])) + sd * rng.uniform(30, 55)
                L = rng.uniform(4, 9)
                cell = cell.union(spike(spine[i] + nrm[i] * sd * w[i] * 0.5, ang, L, L * 0.45,
                                        bend=-sd * 0.3).poly)
            cell = cell.intersection(clip)
            if cell.area > 6:
                out.append(cell)
    return out


def outline(shape, w=LINE):
    return shape.buffer(w / 2, join_style=2, mitre_limit=6).difference(
        shape.buffer(-w / 2, join_style=2, mitre_limit=6))


def scallop(cx, cy, r, rng, k=8, bump=0.42):
    ang = np.linspace(0, 2 * np.pi, k + 1)[:-1] + rng.uniform(0, 1)
    parts = [Point(cx, cy).buffer(r * 0.72, 64)]
    for a in ang:
        rr = r * bump * rng.uniform(0.8, 1.2)
        parts.append(Point(cx + r * 0.72 * math.cos(a), cy + r * 0.72 * math.sin(a)).buffer(rr, 40))
    return unary_union(parts)


# --------------------------------------------------------------------------- #
# compose: silhouette outline + interior lightning + outlined claws
# --------------------------------------------------------------------------- #
class Piece:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.body = []   # filled-and-outlined shapes (get interior lines)
        self.claws = []  # outlined shapes, no interior
        self.solid = []  # always solid (needles, barbs)
        self.holes = []  # inner shapes cut out of the body (outlined)

    def blade(self, P, W, edge=True, **kw):
        b = Blade(P, W, **kw)
        self.body.append(b)
        if edge and W > 20:
            n = int(np.linalg.norm(b.P[3] - b.P[0]) / 55)
            for side in (1, -1):
                self.thorns(b, n, side=side, size=(7, 14), t0=0.15, t1=0.85, back=34, add=False)
        return b

    def claw(self, P, W, solid=False):
        b = Blade(P, W, profile=crescent, fill=False)
        (self.solid if solid else self.claws).append(b.poly)
        return b

    def thorns(self, b, n, side=1, size=(8, 18), t0=0.15, t1=0.85, back=40, add=True):
        """Spikes off the edge of a blade."""
        for t in np.linspace(t0, t1, n) + self.rng.uniform(-0.03, 0.03, n):
            c = b.offset_curve(side * 0.95, t, t + 0.01, n=2)
            dvec = c[1] - c[0]
            ang = math.degrees(math.atan2(dvec[1], dvec[0])) + side * back
            L = self.rng.uniform(*size) * (1.1 - 0.6 * t)
            sp = spike(c[0], ang, L, L * 0.38, bend=-side * 0.35)
            self.claws.append(sp.poly)

    def render(self):
        rng = self.rng
        big = [b for b in self.body if b.fill and b.W >= 22]
        thin = [b.poly for b in self.body if not (b.fill and b.W >= 22)]
        sil = unary_union([b.poly for b in big]) if big else Polygon()
        holes = unary_union(self.holes) if self.holes else Polygon()
        parts = []
        if self.holes:
            parts.append(outline(holes))
        self.claws.extend(thin)
        allc = []
        for b in big:
            allc.extend(cells(b, rng))
        if allc:
            # keep cells separate so each keeps its own outline
            kept, acc = [], Polygon()
            for c in allc:
                c = c.difference(acc.buffer(1.6)).difference(holes.buffer(2.5))
                if c.area > 6:
                    kept.append(c)
                    acc = acc.union(c)
            parts.append(outline(unary_union(kept), FILL_LINE))
        claws = unary_union(self.claws).difference(sil.buffer(-0.01)) if self.claws else Polygon()
        if self.claws:
            cl = unary_union(self.claws)
            parts.append(outline(cl).difference(sil.buffer(-1.0)))
        if self.solid:
            parts.append(unary_union(self.solid))
        g = unary_union(parts)
        g = g.buffer(0.15, join_style=2).buffer(-0.15, join_style=2)
        return g.simplify(0.03)


# --------------------------------------------------------------------------- #
# designs (x = towards the FRONT of the bike, y = down)
# --------------------------------------------------------------------------- #
def side_upper():
    p = Piece(11)
    # main mass: fan of comets heavy at the front (right), sweeping back into a tail
    m1 = p.blade([(600, 70), (470, 105), (300, 175), (40, 160)], 76, rise=0.18, tail=1.3)
    p.blade([(590, 30), (470, 45), (330, 95), (170, 105)], 58, rise=0.25)
    p.blade([(560, 125), (460, 165), (350, 200), (210, 205)], 46, rise=0.25)
    # horns sweeping up and back over the top
    h1 = p.blade([(470, 50), (420, -5), (300, -20), (190, 30)], 24, rise=0.2, tail=1.1)
    h2 = p.blade([(545, 35), (520, -15), (440, -40), (360, -30)], 16, rise=0.2)
    p.thorns(h1, 4, side=-1, size=(10, 16))
    p.thorns(m1, 6, side=1, size=(10, 22), t0=0.3, t1=0.9)
    # front needles shooting forward
    for y, ang, L in ((40, -12, 70), (75, 2, 85), (110, 14, 65)):
        p.claws.append(spike((585, y), ang, L, 10, bend=0.1).poly)
    # claws hanging under the mass
    for x, h, lean in ((470, 95, 30), (400, 115, 25), (330, 110, 18), (265, 90, 12)):
        y = 150 + (470 - x) * 0.18
        p.claw([(x, y), (x - 10, y + h * 0.5), (x - lean - 30, y + h * 0.9),
                (x - lean - 60, y + h * 0.8)], 9)
    # curl at the tail
    p.claw([(130, 150), (70, 215), (0, 200), (10, 140)], 11)
    p.claw([(95, 160), (60, 195), (30, 185), (35, 160)], 6, solid=True)
    # swirl around the eye: curved crescent masses so the cells arc around it
    p.blade([(540, 70), (520, 150), (400, 160), (370, 95)], 34, profile=crescent, edge=False)
    p.blade([(560, 40), (560, 185), (360, 200), (330, 100)], 30, profile=crescent, edge=False)
    # inner eye: a cut-out with its own outline
    p.holes.append(Blade([(500, 75), (470, 60), (430, 70), (400, 95)], 18, profile=crescent).poly)
    return p.render()


def side_lower():
    p = Piece(23)
    b = p.blade([(480, 30), (380, 50), (220, 95), (10, 95)], 54, rise=0.2, tail=1.2)
    p.blade([(470, 70), (400, 95), (300, 120), (160, 125)], 34, rise=0.22)
    p.thorns(b, 7, side=-1, size=(9, 18), t0=0.2, t1=0.9)
    for x, h in ((420, 60), (350, 70), (280, 65)):
        y = 95 + (420 - x) * 0.1
        p.claw([(x, y), (x - 8, y + h * 0.6), (x - 45, y + h), (x - 70, y + h * 0.85)], 8)
    p.claw([(470, 25), (510, -20), (540, -10), (550, 20)], 8)
    p.claws.append(spike((475, 45), 0, 55, 9, bend=0.1).poly)
    return p.render()


def front_cowl():
    p = Piece(37)
    # trunk: heavy at the bottom, needle at the top
    t = p.blade([(150, 360), (135, 260), (100, 130), (40, 10)], 78, rise=0.15, tail=1.15)
    p.blade([(195, 370), (200, 290), (180, 200), (140, 120)], 44, rise=0.2)
    p.thorns(t, 5, side=1, size=(16, 28), t0=0.3, t1=0.85, back=45)
    p.thorns(t, 4, side=-1, size=(14, 24), t0=0.35, t1=0.8, back=45)
    p.claw([(62, 60), (40, 0), (28, -40), (20, -80)], 10)          # top spike
    # tight cluster of thorny scalloped lobes at the base
    for cx, cy, r in ((128, 392, 30), (188, 400, 28), (160, 445, 27), (100, 440, 23), (220, 448, 22)):
        lobe = scallop(cx, cy, r, p.rng, k=int(p.rng.integers(6, 9)), bump=0.3)
        b = Blade([(cx, cy)] * 4, 1, fill=False)
        b.poly = lobe
        p.body.append(b)
        # hooked thorns round the lobe
        for a in np.linspace(0, 2 * np.pi, 7)[:-1] + p.rng.uniform(0, 1):
            q = (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))
            p.claws.append(spike(q, math.degrees(a) + 40, r * p.rng.uniform(0.4, 0.7), r * 0.2,
                                 bend=-0.7).poly)
        # sharp crescent curls inside (cut-outs with their own outline)
        a0 = p.rng.uniform(0, 2 * np.pi)
        for k, (rr, sweep, w) in enumerate(((0.55, 2.4, r * 0.22), (0.28, 2.0, r * 0.14))):
            a = a0 + k * 0.5
            pts = [(cx + rr * r * math.cos(a + sweep * u), cy + rr * r * math.sin(a + sweep * u))
                   for u in (0, 0.33, 0.66, 1)]
            p.holes.append(Blade(pts, w, profile=crescent).poly)
    # long crescent claws framing the piece
    p.claw([(215, 210), (270, 280), (290, 380), (262, 470)], 12)
    p.claw([(70, 250), (25, 320), (20, 400), (45, 465)], 9)
    p.claw([(235, 320), (275, 355), (290, 400), (282, 432)], 7)
    return p.render()


def tank():
    p = Piece(41)
    b = p.blade([(330, 60), (250, 50), (130, 80), (0, 60)], 44, rise=0.2)
    p.blade([(320, 95), (250, 110), (170, 115), (80, 105)], 26, rise=0.25)
    p.thorns(b, 5, side=-1, size=(8, 15), t0=0.2, t1=0.9)
    p.claw([(280, 100), (270, 150), (230, 175), (200, 160)], 7)
    p.claw([(210, 105), (200, 145), (170, 160), (150, 150)], 6)
    p.claws.append(spike((325, 70), 5, 45, 8).poly)
    return p.render()


def tail():
    p = Piece(53)
    b = p.blade([(0, 60), (90, 40), (200, 50), (300, 10)], 40, rise=0.2, tail=1.1)
    p.thorns(b, 5, side=1, size=(8, 16), t0=0.2, t1=0.9)
    p.thorns(b, 4, side=-1, size=(8, 14), t0=0.25, t1=0.8)
    p.claw([(60, 75), (90, 120), (140, 125), (165, 105)], 7)
    p.claws.append(spike((4, 60), 185, 40, 8).poly)
    return p.render()


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def ring_d(coords):
    pts = list(coords)[:-1]
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"


def geom_d(g):
    polys = [g] if g.geom_type == "Polygon" else [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]
    out = []
    for q in polys:
        out.append(ring_d(q.exterior.coords))
        out.extend(ring_d(r.coords) for r in q.interiors)
    return " ".join(out)


def write_svg(path, g, title, color="#ffffff", bg=None, pad=5):
    x0, y0, x1, y1 = g.bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    w, h = x1 - x0, y1 - y0
    bgr = f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{bg}"/>\n' if bg else ""
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{w:.1f}mm" height="{h:.1f}mm" viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}">\n'
                f'<title>{title}</title>\n{bgr}<path fill="{color}" fill-rule="evenodd" d="{geom_d(g)}"/>\n</svg>\n')
    return w, h


def export(name, g, title):
    g = affinity.translate(g, -g.bounds[0], -g.bounds[1])
    os.makedirs(OUT, exist_ok=True)
    dbg = os.path.join(HERE, "preview", "sigil")
    os.makedirs(dbg, exist_ok=True)
    base = os.path.join(OUT, name)
    w, h = write_svg(base + ".svg", g, title)
    write_svg(base + "_black.svg", g, title, color="#000000")
    write_svg(base + "_mirrored.svg", affinity.scale(g, -1, 1, origin="centroid"), title + " (other side)")
    write_svg(os.path.join(dbg, name + ".svg"), g, title, bg="#111111")
    import cairosvg
    cairosvg.svg2pdf(url=base + ".svg", write_to=base + ".pdf")
    cairosvg.svg2png(url=os.path.join(dbg, name + ".svg"), write_to=os.path.join(dbg, name + ".png"),
                     output_width=1800)
    print(f"{name}: {w:.0f} x {h:.0f} mm")


DESIGNS = {
    "sigil_side_upper": (side_upper, "Side fairing sigil"),
    "sigil_side_lower": (side_lower, "Lower fairing sigil"),
    "sigil_front_cowl": (front_cowl, "Front cowl sigil"),
    "sigil_tank": (tank, "Tank sigil"),
    "sigil_tail": (tail, "Tail sigil"),
}

if __name__ == "__main__":
    import sys
    for n in sys.argv[1:] or DESIGNS:
        fn, title = DESIGNS[n]
        export(n, fn(), title)
