"""
Redraw decal line-work from a photo as clean, print-ready vector strokes.

photo -> isolate thin bright lines -> skeleton (centre-lines) -> prune noise ->
corner-preserving smoothing -> re-stroke at a constant width with sharp tapered
ends -> merge into one filled shape -> scale to real size in mm -> SVG / PDF.

Pieces are configured in PIECES at the bottom. Run:  python3 redraw.py [piece ...]
"""
import math
import os
import sys

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from skan import Skeleton, summarize
from skimage.morphology import remove_small_objects, skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
UP = "/root/.claude/uploads/3d723f37-ba52-524f-82f7-87284935da22"
PHOTOS = {
    "gas": f"{UP}/d9d9474a-image.jpg",
    "garage": f"{UP}/3d53a92a-image.jpg",
    "cowl": f"{UP}/2615fb1c-image.jpg",
}
OUT = os.path.join(HERE, "print")
DBG = os.path.join(HERE, "preview", "debug")


# --------------------------------------------------------------------------- #
# 1. raster: isolate the decal lines
# --------------------------------------------------------------------------- #
def line_mask(cfg):
    img = cv2.imread(PHOTOS[cfg["photo"]])
    keep = np.array(cfg["keep"], np.int32)
    x, y, w, h = cv2.boundingRect(keep)
    S = cfg.get("scale", 4)
    crop = img[y:y + h, x:x + w]
    crop = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_LANCZOS4)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), cfg.get("blur", 1.2))
    # white top-hat: thin bright strokes survive, broad paint reflections don't
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.get("tophat", 21),) * 2)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    # local contrast normalisation so dim and bright areas threshold alike
    loc = cv2.GaussianBlur(th, (0, 0), 25) + 4
    r = th / loc
    m = (r > cfg.get("ratio", 2.2)) & (th > cfg.get("floor", 10))
    poly = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(poly, [((keep - [x, y]) * S).astype(np.int32)], 1)
    for ex in cfg.get("exclude", []):
        cv2.fillPoly(poly, [((np.array(ex) - [x, y]) * S).astype(np.int32)], 0)
    m &= poly.astype(bool)
    m = remove_small_objects(m, max_size=cfg.get("min_obj", 40))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    return m, crop


# --------------------------------------------------------------------------- #
# 2. vector: centre-lines -> clean strokes
# --------------------------------------------------------------------------- #
def gauss_smooth(p, sigma):
    if len(p) < 5 or sigma <= 0:
        return p
    k = int(sigma * 3) * 2 + 1
    g = np.exp(-0.5 * (np.arange(k) - k // 2) ** 2 / sigma ** 2)
    g /= g.sum()
    pad = k // 2
    out = np.empty_like(p)
    for i in range(2):
        v = np.r_[np.full(pad, p[0, i]), p[:, i], np.full(pad, p[-1, i])]
        out[:, i] = np.convolve(v, g, "valid")
    out[0], out[-1] = p[0], p[-1]
    return out


def resample(p, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < step:
        return p
    s = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


def corner_smooth(p, sigma, corner_deg=55, probe=6):
    """Smooth a polyline but keep sharp corners (thorn tips) sharp."""
    p = resample(p, 1.0)
    n = len(p)
    if n < 2 * probe + 3:
        return gauss_smooth(p, sigma / 2)
    ang = np.zeros(n)
    for i in range(probe, n - probe):
        a = p[i] - p[i - probe]
        b = p[i + probe] - p[i]
        ca = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        ang[i] = math.degrees(math.acos(np.clip(ca, -1, 1)))
    corners = []
    i = probe
    while i < n - probe:
        if ang[i] > corner_deg:
            j = i
            while j < n - probe and ang[j] > corner_deg:
                j += 1
            corners.append(i + int(np.argmax(ang[i:j])))
            i = j + probe
        else:
            i += 1
    cuts = [0] + corners + [n - 1]
    segs = [gauss_smooth(p[a:b + 1], sigma) for a, b in zip(cuts[:-1], cuts[1:])]
    return np.vstack([segs[0]] + [s[1:] for s in segs[1:]])


def tapered(p, w0, w1):
    """Ribbon along polyline p, width w0 -> w1 (w1=0 gives a needle tip)."""
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(p))
    w = (w0 + (w1 - w0) * t ** 1.3)[:, None] / 2
    return Polygon(np.vstack([p + nrm * w, (p - nrm * w)[::-1]])).buffer(0)


def stroke(p, W, taper_start, taper_end, taper_len):
    L = LineString(p)
    total = L.length
    parts = []
    a, b = 0.0, total
    tl = min(taper_len, total * 0.45)
    if taper_start:
        a = tl
    if taper_end:
        b = total - tl
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

    def sub(s0, s1):
        m = (d >= s0) & (d <= s1)
        return p[m]

    mid = sub(a, b)
    if len(mid) >= 2:
        parts.append(LineString(mid).buffer(W / 2, cap_style=2, join_style=2, mitre_limit=5))
    if taper_start:
        q = sub(0, a + 2.5)[::-1]
        if len(q) >= 2:
            parts.append(tapered(q, W, 0))
    if taper_end:
        q = sub(b - 2.5, total)
        if len(q) >= 2:
            parts.append(tapered(q, W, 0))
    return unary_union(parts)


def bridge_gaps(sk, radius, cone=38):
    """Join broken strokes: extend each loose end forward to the nearest line it points at."""
    from scipy.spatial import cKDTree
    s = Skeleton(sk)
    pts = np.argwhere(sk)
    tree = cKDTree(pts)
    for i in range(s.n_paths):
        idx = s.path(i)
        co = s.path_coordinates(i).astype(float)
        if len(co) < 6:
            continue
        for end, back in ((0, 1), (-1, -1)):
            if s.degrees[idx[end]] != 1:
                continue
            e = co[end]
            j = min(7, len(co) - 1)
            ref = co[j] if end == 0 else co[-1 - j]
            dvec = e - ref
            dvec /= np.linalg.norm(dvec) + 1e-9
            own = {tuple(p) for p in co.astype(int)}
            best = None
            for k in tree.query_ball_point(e, radius):
                q = pts[k]
                if tuple(q) in own:
                    continue
                v = q - e
                dist = np.linalg.norm(v)
                if dist < 1.5:
                    continue
                if np.dot(v / dist, dvec) < math.cos(math.radians(cone)):
                    continue
                if best is None or dist < best[0]:
                    best = (dist, q)
            if best is not None:
                q = best[1]
                img = sk.astype(np.uint8)
                cv2.line(img, (int(e[1]), int(e[0])), (int(q[1]), int(q[0])), 1, 1)
                sk = img.astype(bool)
    return skeletonize(sk)


def vectorize(mask, cfg):
    sk = skeletonize(mask)
    sk = bridge_gaps(sk, cfg.get("bridge", 4 * cfg["line_px"]))
    for _ in range(cfg.get("prune_passes", 3)):
        s = Skeleton(sk)
        df = summarize(s, separator="_")
        drop = df[(df.branch_type == 1) & (df.branch_distance < cfg.get("spur", 7))].index
        drop = list(drop) + list(df[(df.branch_type == 0) & (df.branch_distance < cfg.get("min_len", 9))].index)
        if not drop:
            break
        for i in drop:
            co = s.path_coordinates(i).astype(int)
            # keep junction pixels: only clear interior + endpoint pixels of the spur
            deg = s.degrees[s.path(i)]
            for (r, c), dg in zip(co, deg):
                if dg <= 2:
                    sk[r, c] = False
        sk = skeletonize(sk)
    s = Skeleton(sk)
    df = summarize(s, separator="_")
    W = cfg["line_px"]
    parts = []
    junctions = set()
    for i in range(s.n_paths):
        co = s.path_coordinates(i)[:, ::-1].astype(float)  # (x, y)
        if len(co) < 3:
            continue
        idx = s.path(i)
        d0, d1 = s.degrees[idx[0]], s.degrees[idx[-1]]
        if d0 >= 3:
            junctions.add(tuple(co[0]))
        if d1 >= 3:
            junctions.add(tuple(co[-1]))
        sm = corner_smooth(co, cfg.get("sigma", 3.5))
        parts.append(stroke(sm, W, d0 == 1, d1 == 1, cfg.get("taper_px", 9 * W)))
    for j in junctions:
        parts.append(Point(j).buffer(W * 0.55, 16))
    g = unary_union(parts)
    c = W * 0.25
    g = g.buffer(c, join_style=2, mitre_limit=5).buffer(-c, join_style=2, mitre_limit=5)
    # drop specks / reflection noise
    min_a = cfg.get("min_area", 18) * W * W
    geoms = [q for q in getattr(g, "geoms", [g]) if q.area >= min_a]
    return unary_union(geoms)


# --------------------------------------------------------------------------- #
# 3. output
# --------------------------------------------------------------------------- #
def ring_d(coords):
    pts = list(coords)[:-1]
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"


def geom_d(g):
    polys = [g] if g.geom_type == "Polygon" else [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]
    out = []
    for p in polys:
        out.append(ring_d(p.exterior.coords))
        out.extend(ring_d(r.coords) for r in p.interiors)
    return " ".join(out)


def write_svg(path, shape, title, color="#ffffff", bg=None, pad=5):
    x0, y0, x1, y1 = shape.bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    w, h = x1 - x0, y1 - y0
    bgr = f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{bg}"/>\n' if bg else ""
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" height="{h:.1f}mm" '
           f'viewBox="{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}">\n<title>{title}</title>\n{bgr}'
           f'<path fill="{color}" fill-rule="evenodd" d="{geom_d(shape)}"/>\n</svg>\n')
    with open(path, "w") as f:
        f.write(svg)
    return w, h


LINE_MM = 1.6  # printed line weight, same on every piece


def build(name, cfg):
    keep = np.array(cfg["keep"])
    ext = np.ptp(keep[:, 0]) * cfg.get("stretch", 1) if cfg.get("size_axis", "w") == "w" else np.ptp(keep[:, 1])
    mm_per_px = cfg["size_mm"] / (ext * cfg.get("scale", 4))
    cfg = dict(cfg, line_px=LINE_MM / mm_per_px)
    mask, crop = line_mask(cfg)
    os.makedirs(DBG, exist_ok=True)
    cv2.imwrite(os.path.join(DBG, f"{name}_mask.png"), mask.astype(np.uint8) * 255)
    shape = vectorize(mask, cfg)
    # px -> mm, optional horizontal perspective stretch / rotation
    if cfg.get("rotate"):
        shape = affinity.rotate(shape, cfg["rotate"], origin="centroid")
    if cfg.get("stretch"):
        shape = affinity.scale(shape, cfg["stretch"], 1, origin="centroid")
    if cfg.get("mirror"):
        shape = affinity.scale(shape, -1, 1, origin="centroid")
    x0, y0, x1, y1 = shape.bounds
    k = cfg["size_mm"] / ((x1 - x0) if cfg.get("size_axis", "w") == "w" else (y1 - y0))
    shape = affinity.scale(shape, k, k, origin=(0, 0))
    shape = affinity.translate(shape, -shape.bounds[0], -shape.bounds[1])
    # enforce the real printed line weight after scaling
    shape = shape.simplify(0.05)
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    w, h = write_svg(base + ".svg", shape, cfg["title"])
    write_svg(base + "_black.svg", shape, cfg["title"], color="#000000")
    write_svg(os.path.join(DBG, f"{name}_onblack.svg"), shape, cfg["title"], bg="#111")
    mirrored = affinity.scale(shape, -1, 1, origin="centroid")
    write_svg(base + "_mirrored.svg", mirrored, cfg["title"] + " (other side)")
    try:
        import cairosvg
        cairosvg.svg2pdf(url=base + ".svg", write_to=base + ".pdf")
        cairosvg.svg2png(url=os.path.join(DBG, f"{name}_onblack.svg"),
                         write_to=os.path.join(DBG, f"{name}.png"), output_width=1800)
    except Exception as e:  # noqa
        print("pdf/png skipped:", e)
    print(f"{name}: {w:.0f} x {h:.0f} mm  ({len(getattr(shape, 'geoms', [shape]))} parts)")


# Coordinates are pixels in the original photos. Sizes are real-world mm for a
# Yamaha R7 / R9 / R1-size fairing (front wheel 120/70-17 ~ 600 mm used as ruler).
PIECES = {
    "side_upper": dict(
        title="Side fairing sigil", photo="garage", size_mm=620,
        keep=[(55, 610), (130, 575), (220, 552), (300, 540), (370, 538), (425, 548),
              (428, 566), (385, 590), (372, 612), (330, 622), (300, 640), (275, 690),
              (205, 700), (175, 712), (55, 705)],
        line_px=5.0, tophat=19, ratio=2.0, stretch=1.12,
    ),
    "side_lower": dict(
        title="Lower fairing sigil", photo="gas", size_mm=470,
        keep=[(234, 668), (270, 662), (300, 668), (318, 655), (345, 636), (375, 628),
              (406, 632), (406, 650), (386, 668), (370, 690), (356, 714), (330, 719),
              (290, 721), (250, 719), (232, 702)],
        exclude=[[(294, 668), (332, 668), (332, 700), (294, 700)]],
        line_px=5.5, tophat=21, ratio=1.8, stretch=1.3, scale=5,
    ),
    "front_cowl": dict(
        title="Front cowl sigil", photo="cowl", size_mm=420, size_axis="h",
        keep=[(115, 200), (190, 260), (260, 360), (300, 420), (360, 470), (395, 560),
              (385, 680), (330, 680), (250, 670), (195, 640), (175, 560), (150, 430),
              (120, 300)],
        exclude=[[(290, 250), (450, 250), (450, 400), (300, 400)]],
        line_px=4.5, tophat=17, ratio=2.0, scale=4,
    ),
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(PIECES)
    for n in names:
        build(n, PIECES[n])
