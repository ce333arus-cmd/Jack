# Cybersigil fairing decals – vector files

Line-art redrawn from photos of the original decals (no text).
Everything in `print/` is real size in **millimetres**. The line weight is **1.6 mm** on every piece.

| Piece | File | Size (mm) | Where |
|---|---|---|---|
| Side sigil (upper) | `print/side_upper*` | 630 × 263 | mid fairing, each side |
| Side sigil (lower) | `print/side_lower*` | 480 × 202 | lower fairing / belly, each side |
| Front cowl sigil | `print/front_cowl*` | 271 × 430 | nose, beside the headlight / screen |

For each piece:
- `.svg` – white artwork (for black or dark bikes), vector
- `_black.svg` – the same artwork in black (for light-coloured bikes, or to send to a printer)
- `_mirrored.svg` – flipped copy for the other side of the bike
- `.pdf` – vector PDF, the format most print shops ask for

## Sizing / universal fit
Sizes are set for Yamaha R7 / R9 / R1-class fairings (front wheel ≈ 600 mm used as the ruler).
Because the files are vector, they scale with no loss of quality. Roughly:
- 600–1000 cc supersports (R6, ZX-6R/10R, CBR600/1000, GSX-R, Panigale V2/V4, S1000RR): 90–110 %
- 300–400 cc (R3, Ninja 400, RC390): 70–80 %

Keep the line weight at **1.2 mm or more** after scaling if you're cutting vinyl, so it can be weeded.

## Regenerating
`python3 redraw.py` (needs: opencv-python-headless, scikit-image, skan, shapely, scipy, cairosvg).
Each piece is set up in `PIECES` at the bottom of `redraw.py`: which photo it comes from, the
outline of the area to trace, real size and perspective stretch. Swap in higher-resolution photos
there for a cleaner trace.

`old_concept/` holds the first generated designs and the lettering (not used).
