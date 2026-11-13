"""Face fixes for the v7 build: dot eyes and two key scales.

This module is called from :mod:`build` through three small hooks:

* ``face_shapes(index, selected)`` replaces ring eyes with dots before the
  figure is written to the film;
* ``extra_transform(index, f)`` adds a similarity correction for the two
  badly scaled in-between keys;
* ``validate_mouths()`` fails the build if a used key has no mouth at all.

Mouth interiors keep the colours they were fitted with.  The source drawings
(``sources/``) are only read here.  ``ncel/sources/`` is never changed.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.affinity import affine_transform
from shapely.geometry import Point, box
from shapely.ops import unary_union

SKIN = (252, 200, 158)
INK = "#111416"
WHITE = "#FEFEFE"

# The two in-betweens drawn smaller than their neighbours.  They are key ids
# in process/polish/extra_keys.json (102 = frames 95-99, 105 = 108-109).
NORMALISE_KEY_IDS = {102, 105}

_B = None


def configure(build_module):
    """Give this module access to build's inventory/geometry helpers."""
    global _B
    _B = build_module


# ---------------------------------------------------------------------------
# source drawings and head boxes


def _source_path(index):
    key = _B.INV[index]
    if key["id"] < 100:
        return _B.ROOT / f"sources/cels/f{key['id']:04d}.jpg"
    return _B.ROOT / "sources/extras" / Path(key["_extra_src"]).name


@lru_cache(maxsize=None)
def _source_rgb(index):
    return np.asarray(Image.open(_source_path(index)).convert("RGB"))


def _largest_component(mask):
    lab, _n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    best = None
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if best is None or area > best[0]:
            best = (area, sl)
    return best


@lru_cache(maxsize=None)
def _source_head(index):
    """Head box in canvas coordinates, plus the source's head kind.

    A simple detector: skin for a painted
    face, otherwise the largest white hole near the top of the ink figure.
    It is only used to locate the lower face; it does not alter the drawing.
    """
    a = _source_rgb(index).astype(np.int16)
    skin = np.abs(a - np.array(SKIN)).max(2) < 45
    best = _largest_component(skin)
    kind = None
    bbox = None
    if best is not None and best[0] >= 1500:
        sl = best[1]
        kind = "skin"
        bbox = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
    else:
        ink = a.sum(2) < 300
        best = _largest_component(ink)
        if best is None:
            return None, None
        sl = best[1]
        ys, ye = sl[0].start, sl[0].stop
        xs, xe = sl[1].start, sl[1].stop
        sub = ink[max(0, ys - 2):min(ink.shape[0], ye + 2),
                  max(0, xs - 2):min(ink.shape[1], xe + 2)]
        holes = ndimage.binary_fill_holes(sub) & ~sub
        labh, _nh = ndimage.label(holes)
        candidates = []
        for i, sh in enumerate(ndimage.find_objects(labh), 1):
            if sh is None:
                continue
            area = int((labh[sh] == i).sum())
            if area > 1000 and sh[0].start < 0.20 * (ye - ys) + 2:
                candidates.append((area, sh))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _area, sh = candidates[0]
            kind = "hole"
            bbox = (sh[1].start + xs, sh[0].start + ys,
                    sh[1].stop + xs, sh[0].stop + ys)
        else:
            kind = "figuretop"
            bbox = (xs, ys, xe, ys + max(20, int((ye - ys) * 0.35)))
    if bbox is None:
        return None, None
    s, tx, ty = _B._mapping(index)
    return (bbox[0] * s + tx, bbox[1] * s + ty,
            bbox[2] * s + tx, bbox[3] * s + ty), kind


@lru_cache(maxsize=None)
def _source_open_mouth(index):
    """True if the source drawing's lower face encloses a non-ink hole.

    This is deliberately independent of the traced shapes: it lets the build
    tell a genuinely closed-line mouth from an open mouth whose interior fill
    has disappeared from the trace.
    """
    head_bbox, _kind = _source_head(index)
    if head_bbox is None:
        return False
    s, tx, ty = _B._mapping(index)
    x0, y0 = (head_bbox[0] - tx) / s, (head_bbox[1] - ty) / s
    x1, y1 = (head_bbox[2] - tx) / s, (head_bbox[3] - ty) / s
    a = _source_rgb(index).astype(np.int16)
    ink = a.sum(2) < 300
    X0, Y0 = max(0, int(x0) - 2), max(0, int(y0) - 2)
    X1, Y1 = min(ink.shape[1], int(x1) + 3), min(ink.shape[0], int(y1) + 3)
    if X1 <= X0 or Y1 <= Y0:
        return False
    crop = ink[Y0:Y1, X0:X1]
    lab, _n = ndimage.label(crop, structure=np.ones((3, 3), bool))
    lower = y0 + 0.55 * (y1 - y0)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < 60:
            continue
        sub = lab[sl] == i
        holes = ndimage.binary_fill_holes(sub) & ~sub
        labh, _nh = ndimage.label(holes)
        for j, sh in enumerate(ndimage.find_objects(labh), 1):
            if sh is None:
                continue
            ha = int((labh[sh] == j).sum())
            if ha < 80:
                continue
            cx = X0 + sl[1].start + 0.5 * (sh[1].start + sh[1].stop)
            cy = Y0 + sl[0].start + 0.5 * (sh[0].start + sh[0].stop)
            if cy < lower:
                continue
            if x0 <= cx <= x1:
                return True
    return False


def _face_geometry(index, paints, head_bbox, head_kind):
    """Largest skin fill covering the detected head (painted keys only)."""
    if head_kind != "skin":
        return None
    hbb = box(*head_bbox)
    skin = [(n, c, g) for n, c, g in paints
            if c == "#FCC89E" and g.area > 2000]
    if not skin:
        return None
    skin.sort(key=lambda t: (t[2].intersection(hbb).area, t[2].area),
              reverse=True)
    return skin[0][2]


def _fills_inside(ink, paints):
    out = []
    for n, c, g in paints:
        if c == INK or g.is_empty or g.area < 25:
            continue
        if g.area >= 0.95 * ink.area:
            continue
        rep = g.representative_point()
        if ink.buffer(0.5).contains(rep) and \
                g.intersection(ink).area > 0.75 * g.area:
            out.append((n, c, g))
    # The source stack often carries the same fill twice (outer + inner
    # trace).  Keep one record per colour/area/box.
    seen = set()
    unique = []
    for rec in out:
        sig = (rec[1], round(rec[2].area, 1), rec[2].bounds)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(rec)
    return unique


# ---------------------------------------------------------------------------
# mouth


def _enclosing_mouth(index, paints, head_bbox, head_kind):
    """Return (kind, ink_name, fills, face) for the key's mouth."""
    face = _face_geometry(index, paints, head_bbox, head_kind)
    region = face.bounds if face is not None else head_bbox
    x0, y0, x1, y1 = region
    lower = y0 + 0.60 * (y1 - y0)
    inks = []
    for n, c, g in paints:
        if c != INK or g.is_empty or not (20 <= g.area <= 12000):
            continue
        b = g.bounds
        if b[0] < head_bbox[0] - 8 or b[2] > head_bbox[2] + 8 or \
                b[1] < head_bbox[1] - 8 or b[3] > head_bbox[3] + 8:
            continue
        if 0.5 * (b[1] + b[3]) < lower:
            continue
        if face is not None and g.intersection(face).area < 0.80 * g.area:
            continue
        inks.append((n, g))

    interior = []
    for n, g in inks:
        fills = _fills_inside(g, paints)
        if fills:
            interior.append((n, g, fills))
    if interior:
        # The mouth is the lowest nested shape in the face.  This deliberately
        # rejects ring eyes, which sit above it.
        n, g, fills = max(interior, key=lambda t: t[1].bounds[1] + t[1].bounds[3])
        return "interior", n, fills, face

    lines = []
    for n, g in inks:
        b = g.bounds
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 8 or h < 1 or w < 1.0 * h or g.area < 20 or g.area > 2000:
            continue
        lines.append((n, g))
    if lines:
        n, g = max(lines, key=lambda t: t[1].area)
        return "line", n, [], face
    return "none", None, [], face


@lru_cache(maxsize=None)
def mouth_info(index):
    """The mouth ink and its fitted interior fills."""
    head_bbox, head_kind = _source_head(index)
    if head_bbox is None:
        return {"index": index, "kind": "none", "fills": []}
    key = _B.INV[index]
    paints = _B.raw(index)
    kind, ink_name, fills, _face = _enclosing_mouth(
        index, paints, head_bbox, head_kind)
    records = [{"name": n, "area": round(g.area, 1), "current": c}
               for n, c, g in fills]
    return {
        "index": index,
        "key_id": key["id"],
        "frame": key["frame"],
        "head_kind": head_kind,
        "kind": kind,
        "ink": ink_name,
        "fills": records,
    }


# ---------------------------------------------------------------------------
# eyes


@lru_cache(maxsize=None)
def _eye_info(index):
    head_bbox, head_kind = _source_head(index)
    if head_bbox is None or head_kind != "skin":
        return {"rings": [], "dot_radius": None}
    key = _B.INV[index]
    paints = _B.raw(index)
    face = _face_geometry(index, paints, head_bbox, head_kind)
    if face is None:
        return {"rings": [], "dot_radius": None}
    fbb = face.bounds
    fh = fbb[3] - fbb[1]
    whites = []
    for n, c, g in paints:
        if c != WHITE or not (250 <= g.area <= 5000):
            continue
        b = g.bounds
        w, h = b[2] - b[0], b[3] - b[1]
        cy = 0.5 * (b[1] + b[3])
        if cy < fbb[1] + 0.04 * fh or cy > fbb[1] + 0.62 * fh:
            continue
        if w > 0.30 * (fbb[2] - fbb[0]) or h > 0.30 * fh:
            continue
        if g.area < 0.30 * w * h:
            continue
        if not face.buffer(1).contains(g.representative_point()):
            continue
        whites.append((n, c, g))

    groups = []
    for rec in whites:
        cx, cy = rec[2].centroid.x, rec[2].centroid.y
        for grp in groups:
            if any(abs(cx - q[2].centroid.x) < 10 and
                   abs(cy - q[2].centroid.y) < 10 for q in grp):
                grp.append(rec)
                break
        else:
            groups.append([rec])

    rings = []
    for grp in groups:
        wu = unary_union([q[2] for q in grp])
        inks = []
        for n, c, g in paints:
            if c != INK or g.is_empty or g.area > 10000:
                continue
            if g.distance(wu) < 4 or not g.intersection(wu.buffer(2)).is_empty:
                inks.append((n, g))
        pupils = []
        for n, c, g in paints:
            if c != INK or g.area < 5 or g.area >= 0.7 * wu.area:
                continue
            if wu.buffer(1).contains(g.representative_point()):
                pupils.append((n, g))
        centre = max(pupils, key=lambda t: t[1].area)[1].centroid \
            if pupils else wu.centroid
        rings.append({
            "white": [q[0] for q in grp],
            "inks": [n for n, _g in inks],
            "pupils": [n for n, _g in pupils],
            "centre": (centre.x, centre.y),
            "white_area": wu.area,
        })

    dots = _dot_pair(paints, face, fbb, fh)
    dot_radius = None
    if dots:
        dot_radius = math.sqrt((dots[0].area + dots[1].area) / 2 / math.pi)
    return {
        "index": index,
        "key_id": key["id"],
        "frame": key["frame"],
        "head_height": head_bbox[3] - head_bbox[1],
        "face_height": fh,
        "rings": rings,
        "dot_radius": dot_radius,
    }


def _dot_pair(paints, face, fbb, fh):
    dots = []
    for n, c, g in paints:
        if c != INK or not (50 <= g.area <= 900):
            continue
        b = g.bounds
        w, h = b[2] - b[0], b[3] - b[1]
        cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        if w < 8 or h < 8 or not (0.55 < w / h < 1.8):
            continue
        if cy < fbb[1] + 0.05 * fh or cy > fbb[1] + 0.55 * fh:
            continue
        if not face.buffer(1).contains(g.representative_point()):
            continue
        dots.append((n, g))
    best = None
    for i, (n1, g1) in enumerate(dots):
        b1 = g1.bounds
        for n2, g2 in dots[i + 1:]:
            b2 = g2.bounds
            if max(g1.area, g2.area) / min(g1.area, g2.area) > 1.6:
                continue
            c1x, c1y = 0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])
            c2x, c2y = 0.5 * (b2[0] + b2[2]), 0.5 * (b2[1] + b2[3])
            if abs(c1y - c2y) > 0.08 * fh:
                continue
            if abs(c1x - c2x) < 0.12 * (fbb[2] - fbb[0]):
                continue
            score = min(g1.area, g2.area)
            if best is None or score > best[0]:
                best = (score, g1, g2)
    return [best[1], best[2]] if best else []


@lru_cache(maxsize=None)
def _nearest_dot(index):
    """Nearest key in film order that still has a pair of dot eyes."""
    usable = [k["index"] for k in _B.USABLE]
    pos = usable.index(index)
    here = _B.INV[index]["frame"]
    candidates = []
    for other in usable:
        if other == index:
            continue
        info = _eye_info(other)
        if info["rings"] or info["dot_radius"] is None:
            continue
        candidates.append((abs(_B.INV[other]["frame"] - here),
                           _B.INV[other]["frame"], other, info))
    if not candidates:
        return None
    return min(candidates, key=lambda t: (t[0], t[1]))


def _eye_shapes(selected, info):
    if not info["rings"]:
        return selected
    idx = info["index"]
    # Size from the nearest dot-eyed key, scaled by head height.
    nearest = _nearest_dot(idx)
    if nearest is None:
        raise RuntimeError(f"no dot-eyed key to size the eyes for key {idx}")
    _dist, _frame, _other, other_info = nearest
    ratio = info["head_height"] / other_info["head_height"]
    radius = other_info["dot_radius"] * ratio
    remove = set()
    for ring in info["rings"]:
        remove.update(ring["white"])
        remove.update(ring["inks"])
        remove.update(ring["pupils"])
    out = [(n, c, g) for n, c, g in selected if n not in remove]
    for i, ring in enumerate(info["rings"]):
        cx, cy = ring["centre"]
        out.append((f"face_dot_{idx}_{i}", INK,
                    Point(cx, cy).buffer(radius, quad_segs=12)))
    return out


# ---------------------------------------------------------------------------
# the two scale-corrected in-betweens


def _path_value(points, f):
    if f <= points[0][0]:
        return points[0][1]
    for (a, x), (b, y) in zip(points, points[1:]):
        if f < b:
            u = (f - a) / (b - a)
            u = 0.5 * u + 0.5 * (u * u * (3 - 2 * u))
            return x + (y - x) * u
    return points[-1][1]


def _smoothstep(a, b, x):
    if b == a:
        return 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _transformed(index, f, geom):
    _sel, _mask, hb = _B.parts(index)
    hx, hy = 0.5 * (hb[0] + hb[2]), 0.5 * (hb[1] + hb[3])
    hh = hb[3] - hb[1]
    fin = _smoothstep(40, 52, f) * (1 - _smoothstep(92, 100, f))
    bx = (3.0 * math.sin(f * 0.09) + 1.5 * math.sin(f * 0.21) +
          14 * math.sin((f - 40) * 0.105) * fin)
    by = (2.0 * math.sin(f * 0.07 + 1.1) + 1.0 * math.sin(f * 0.17) +
          12 * math.sin((f - 40) * 0.083 + 0.7) * fin)
    tx = _path_value(_B.X, f) + bx + 2.0 * math.sin(f * 0.13)
    ty = _path_value(_B.Y, f) + by + 1.5 * math.sin(f * 0.17 + 0.5)
    sz = _path_value(_B.S, f) * (1 + 0.009 * math.sin(f * 0.33))
    ang = _path_value(_B.ANGLE, f) + 0.55 * math.sin(f * 0.27)
    k = sz / hh
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    a, b, d, e = k * c, -k * s, k * s, k * c
    return affine_transform(geom, [a, b, d, e,
                                   tx - (a * hx + b * hy),
                                   ty - (d * hx + e * hy)])


def _main_figure(index):
    _sel, _mask, _hb = _B.parts(index)
    inks = [(n, c, g) for n, c, g in _sel if c == INK]
    if not inks:
        return None
    return max(inks, key=lambda t: t[2].area)[2]


@lru_cache(maxsize=None)
def _normalisation(index):
    key = _B.INV[index]
    if key["id"] not in NORMALISE_KEY_IDS:
        return None
    usable = [k["index"] for k in _B.USABLE]
    pos = usable.index(index)
    if pos == 0 or pos == len(usable) - 1:
        return None
    prev_i, next_i = usable[pos - 1], usable[pos + 1]
    prev_f = _B.INV[prev_i]["frame"]
    next_f = _B.INV[next_i]["frame"]
    geom = _main_figure(index)
    if geom is None:
        return None

    def bounds_at(i, f):
        g = _main_figure(i)
        b = _transformed(i, f, g).bounds
        return b

    pb = bounds_at(prev_i, prev_f)
    nb = bounds_at(next_i, next_f)
    target_h = 0.5 * ((pb[3] - pb[1]) + (nb[3] - nb[1]))

    f0 = key["frame"]
    f1 = min(next_f - 1, f0 + 4)
    if f1 <= f0:
        f1 = f0
    cur0 = bounds_at(index, f0)
    cur1 = bounds_at(index, f1)
    cur_h = 0.5 * ((cur0[3] - cur0[1]) + (cur1[3] - cur1[1]))
    k = target_h / cur_h
    anchor_x = 0.5 * (cur0[0] + cur0[2])
    anchor_y = cur0[3]

    def target_feet(f):
        u = (f - prev_f) / (next_f - prev_f)
        return pb[3] + (nb[3] - pb[3]) * u

    def correction_at(f, cur):
        scaled_feet = anchor_y + k * (cur[3] - anchor_y)
        return target_feet(f) - scaled_feet

    dy0 = correction_at(f0, cur0)
    dy1 = correction_at(f1, cur1)
    slope = (dy1 - dy0) / (f1 - f0) if f1 != f0 else 0.0
    return {
        "index": index,
        "key_id": key["id"],
        "frame": f0,
        "prev": (prev_i, prev_f, pb),
        "next": (next_i, next_f, nb),
        "before": (cur0[3] - cur0[1], cur0[3]),
        "after": (k * (cur0[3] - cur0[1]), cur0[3] + dy0),
        "scale": k,
        "anchor": (anchor_x, anchor_y),
        "dy0": dy0,
        "slope": slope,
        "f0": f0,
    }


def extra_transform(index, f):
    """Extra film transform for a scale-corrected in-between, or ''."""
    info = _normalisation(index)
    if info is None:
        return ""
    k = info["scale"]
    ax, ay = info["anchor"]
    dy = info["dy0"] + info["slope"] * (f - info["f0"])
    return (f"    transform figure(anchor: ({ax:.3f}, {ay:.3f}), "
            f"scale: ({k:.6f}, {k:.6f}), "
            f"translate: (0, {dy:.3f}))\n")


# ---------------------------------------------------------------------------
# public build hooks


def face_shapes(index, selected):
    """Replace ring eyes with dots in a figure key."""
    eyes = _eye_info(index)
    if eyes["rings"]:
        selected = _eye_shapes(selected, eyes)
    return selected


def validate_mouths():
    """Fail loudly if any used key after the first drawn head lost its mouth."""
    usable = [k for k in _B.USABLE]
    first_head = None
    for k in usable:
        hb, _kind = _source_head(k["index"])
        if hb is not None:
            first_head = k["frame"]
            break
    if first_head is None:
        raise RuntimeError("no drawn head found in any used source key")
    missing = []
    lost_interior = []
    for k in usable:
        if k["frame"] < first_head:
            continue
        hb, _kind = _source_head(k["index"])
        if hb is None:
            continue
        info = mouth_info(k["index"])
        where = (f"key {k['id']} at frame {k['frame']} "
                 f"(source {_source_path(k['index']).name})")
        if info["kind"] == "none":
            missing.append(where)
        elif info["kind"] != "interior" and _source_open_mouth(k["index"]):
            lost_interior.append(where)
    if missing or lost_interior:
        parts = []
        if missing:
            parts.append("no detectable mouth shape in " + ", ".join(missing))
        if lost_interior:
            parts.append("open mouth has no interior fill in " +
                         ", ".join(lost_interior))
        raise RuntimeError("; ".join(parts))


def report():
    """Return the measured numbers used in process/v7/reports/d-face.md."""
    mouth_rows = []
    for k in _B.USABLE:
        info = mouth_info(k["index"])
        mouth_rows.append(info)
    eye_rows = []
    for k in _B.USABLE:
        info = _eye_info(k["index"])
        if info["rings"]:
            eye_rows.append(info)
    norm_rows = [info for k in _B.USABLE
                 if (info := _normalisation(k["index"])) is not None]
    return {"mouths": mouth_rows, "eyes": eye_rows, "normalise": norm_rows}
