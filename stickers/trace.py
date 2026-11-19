"""
Trace the white decal line-work straight out of a photo into smooth vector outlines.

    python3 trace.py <photo> <x> <y> <w> <h> <out_name> [--scale 4] [--tophat 25] [--thresh 0]

Pipeline: crop -> upscale -> white top-hat (removes paint reflections, keeps thin
bright lines) -> threshold -> despeckle -> contours -> curve smoothing -> SVG.
Output: svg/<out_name>_trace.svg (filled white line-work, black parts are holes)
"""
import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def smooth_contour(c, sigma):
    c = c[:, 0, :].astype(float)
    if len(c) < 8:
        return c
    # resample evenly then circular gaussian smoothing -> clean curves, sharp-ish tips kept
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1))]
    n = max(8, int(d[-1] / 1.5))
    s = np.linspace(0, d[-1], n, endpoint=False)
    cc = np.vstack([c, c[:1]])
    x = np.interp(s, d, cc[:, 0])
    y = np.interp(s, d, cc[:, 1])
    k = int(sigma * 3) * 2 + 1
    g = np.exp(-0.5 * (np.arange(k) - k // 2) ** 2 / sigma ** 2)
    g /= g.sum()
    pad = k // 2
    xs = np.convolve(np.r_[x[-pad:], x, x[:pad]], g, "valid")
    ys = np.convolve(np.r_[y[-pad:], y, y[:pad]], g, "valid")
    return np.stack([xs, ys], 1)


def to_path(pts):
    # Catmull-Rom -> cubic bezier for smooth curves
    n = len(pts)
    out = [f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"]
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6
        c2 = p2 - (p3 - p1) / 6
        out.append(f"C{c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return "".join(out) + "Z"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("x", type=int); ap.add_argument("y", type=int)
    ap.add_argument("w", type=int); ap.add_argument("h", type=int)
    ap.add_argument("name")
    ap.add_argument("--scale", type=float, default=4)
    ap.add_argument("--tophat", type=int, default=25)
    ap.add_argument("--thresh", type=int, default=0, help="0 = Otsu")
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--minarea", type=float, default=60)
    a = ap.parse_args()

    img = cv2.imread(a.photo)[a.y:a.y + a.h, a.x:a.x + a.w]
    img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_LANCZOS4)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 30, 7)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.tophat, a.tophat))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    if a.thresh:
        _, bw = cv2.threshold(th, a.thresh, 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cv2.imwrite(os.path.join(HERE, "preview", f"{a.name}_mask.png"), bw)

    cs, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    paths = []
    for c in cs:
        if abs(cv2.contourArea(c)) < a.minarea:
            continue
        paths.append(to_path(smooth_contour(c, a.sigma)))
    H, W = bw.shape
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
           f'<path fill="#ffffff" fill-rule="evenodd" d="{" ".join(paths)}"/>\n</svg>\n')
    out = os.path.join(HERE, "svg", f"{a.name}_trace.svg")
    open(out, "w").write(svg)
    print(out, len(paths), "shapes")


if __name__ == "__main__":
    main()
