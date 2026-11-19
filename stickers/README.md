# Bike decals – vector recreation

Cybersigil / neo-tribal outline decals plus the windscreen and "NO STEP" lettering,
rebuilt as vectors from photos. All files are in **millimetres** at real size.

| File | Piece | Size (mm) |
|---|---|---|
| `svg/01_side_fairing_*` | big sigil on the mid fairing | 660 × 298 |
| `svg/02_lower_fairing_*` | long claw piece under it | 607 × 97 |
| `svg/03_front_cowl_*` | tall piece on the nose / cowl | 258 × 489 |
| `svg/04_tail_tank_*` | small piece for tail / tank | 300 × 167 |
| `svg/05_windscreen_text.svg` | GOTH GF / RAPID / RESPONSE / UNIT | 140 × 90 |
| `svg/06_no_step.svg` | NO STEP | 48 × 18 |

Each sigil comes in three versions:
- `_cut.svg` – **use this for vinyl cutting / printing.** The 1.6 mm white outline is already
  converted to filled shapes.
- `_stroke.svg` – the outline as an editable stroke (change the line weight in Illustrator / Inkscape).
- `_solid.svg` – filled silhouette.

Mirror the side / lower pieces horizontally for the other side of the bike.
Scale in your cutter software to fit the panel. Line weight and sizes are set at the
top of `generate.py` (`LINE`), and the designs are under "designs" in the same file.

Regenerate: `pip install shapely fonttools brotli numpy && python3 generate.py`
Previews: `node preview.js` (Playwright) → `preview/`.

Font: UnifrakturMaguntia (SIL Open Font License, see `fonts/`). Text is already converted to outlines.
