"""Figure and head extraction for the keys.

* the face/head is found by detection, not a fixed table of shape numbers;
* extras are native 1536x1024 drawings whose actor is cut out from the
  drawing itself (see build.extra_actor).
"""
import math

import numpy as np
from shapely.geometry import Polygon, GeometryCollection, box
from shapely.ops import unary_union

COL = {'#FCC89E', '#5A3A26', '#EE2C2C', '#1279D0'}
WHITE = '#FEFEFE'
GREENS = {'#6FDA3A', '#73F203', '#4B9A32'}
SCENERY = {'#03B9FD', '#5CD1FC', '#BFE6F4'} | GREENS
BG = SCENERY | {'#F7941D'}


def polygons(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if hasattr(g, 'geoms'):
        return [p for q in g.geoms for p in polygons(q)]
    return []


def _union(gs):
    gs = [g for g in gs if g is not None and not g.is_empty]
    return unary_union(gs) if gs else GeometryCollection()


def actor_paints(paints):
    """Cut the figure out of a coloured key (v6parts.actor_paints)."""
    coloured = [(n, c, g) for n, c, g in paints if c in COL and g.area > 2500]
    if not coloured:
        return paints, None
    core = _union([g for _, _, g in coloured])
    whites = [(n, c, g) for n, c, g in paints
              if c == WHITE and 500 < g.area < 30000 and g.intersects(core.buffer(70))]
    core2 = _union([core] + [g for _, _, g in whites])
    buf = core2.buffer(22)
    out = []
    for n, c, g in paints:
        if c in BG:
            continue
        if c == WHITE:
            if g.area > 30000:
                continue
            if g.intersects(core.buffer(70)):
                out.append((n, c, g))
            continue
        if c == '#111416':
            q = g.intersection(buf)
            if not q.is_empty and q.area > 0.5:
                out.append((n, c, q))
            continue
        if g.intersects(buf):
            out.append((n, c, g))
    return out, core2


def face_box(paints, fills_only=True):
    """BBox of the face/head/body fill used to split head from body.

    v6 used the hand-written FACE table for the reference-key shape number.
    Shape order moved when the keys were re-fitted, so we detect the fill:
    the largest white (paper keys) or skin (coloured keys) oval in the upper
    part of the frame.  Ink is ignored so the outer contour cannot win.

    `fills_only=False` restores the v6parts rule (any colour, skin boosted);
    it is kept for the five cloud-pose cels the v4 actor rule cannot split.
    """
    cands = []
    for n, c, g in paints:
        if g.is_empty:
            continue
        if fills_only and c not in (WHITE, '#FCC89E'):
            continue
        if c == '#111416' and fills_only:
            continue
        x0, y0, x1, y1 = g.bounds
        w, h = x1 - x0, y1 - y0
        if not (120 <= w <= 460 and 120 <= h <= 480):
            continue
        cy = (y0 + y1) / 2
        if cy > 0.60 * 1080:
            continue
        aspect = w / max(1.0, h)
        if not (0.5 <= aspect <= 1.6):
            continue
        score = g.area * (1.0 - 0.30 * cy / 1080)
        if c == '#FCC89E':
            score *= 1.4
        cands.append((score, c, g.bounds))
    if not cands:
        u = _union([g for _, _, g in paints])
        b = u.bounds
        h = 0.24 * (b[3] - b[1])
        return [b[0], b[1], b[2], b[1] + h]
    cands.sort(key=lambda t: -t[0])
    return list(cands[0][2])


def parts5(paints, fills_only=True):
    fb = face_box(paints, fills_only=fills_only)
    headmask = box(fb[0] - 42, fb[1] - 72, fb[2] + 42, fb[3] + 8)
    headpieces = []
    for n, c, g in paints:
        q = g.intersection(headmask)
        if not q.is_empty and q.area > 0.2:
            headpieces.append(q)
    hb = _union(headpieces).bounds
    if not hb or len(hb) != 4 or any(not np.isfinite(v) for v in hb):
        hb = (fb[0] - 42, fb[1] - 72, fb[2] + 42, fb[3] + 8)
    return paints, headmask, hb


def parts_v4(paints, index):
    """v6 _parts_v4, with the face box detected rather than looked up."""
    fb = face_box(paints)
    x0, y0, x1, y1 = fb
    headmask = box(x0 - 42, y0 - 72, x1 + 42, y1 + 8)
    if index >= 76:
        headmask = box(650, 100, 1150, y1 + 8)
    if index < 34:
        selected = paints
    elif index < 76:
        candidates = [(n, c, g) for n, c, g in paints
                      if c == '#111416' and g.bounds[0] > 300 and
                      g.bounds[2] < 1550 and g.area > 100000]
        if candidates:
            candidates = [q for q in candidates
                          if q[2].intersection(box(x0, y0, x1, y1)).area >
                          0.3 * (x1 - x0) * (y1 - y0)]
        if not candidates:
            # fall back to the largest ink silhouette (some AI keys have a
            # smaller body outline than the 100k threshold used by v6)
            silhouettes = [q for q in paints if q[1] == '#111416' and
                           q[2].area > 20000 and
                           q[2].intersection(box(x0, y0, x1, y1)).area >
                           0.3 * (x1 - x0) * (y1 - y0)]
            if not silhouettes:
                silhouettes = [q for q in paints if q[1] == '#111416']
            if not silhouettes:
                raise ValueError(f'no actor silhouette for {index}')
            candidates = silhouettes
        main = max(candidates, key=lambda q: q[2].area)[2]
        actor = main.buffer(1)
        selected = [(n, c, g.intersection(actor)) for n, c, g in paints
                    if c not in SCENERY]
    else:
        materials = []
        for n, c, g in paints:
            bx = g.bounds
            if not bx:
                continue
            if c in ['#EE2C2C', '#1279D0', '#FCC89E', '#5A3A26'] and \
                    bx[0] > 450 and bx[2] < 1450 and g.area > 1000:
                materials.append(g)
            if c == WHITE and bx[0] > 500 and bx[2] < 1250 and \
                    bx[1] > 880 and g.area > 1000:
                materials.append(g)
        silhouettes = [g for _, c, g in paints if c == '#111416' and
                       g.intersection(box(x0, y0, x1, y1)).area >
                       0.3 * (x1 - x0) * (y1 - y0)]
        if not silhouettes:
            silhouettes = [g for _, c, g in paints if c == '#111416']
        if not silhouettes:
            raise ValueError(f'no silhouettes for {index}')
        main = max(silhouettes, key=lambda g: g.area)
        actual_head = main.intersection(headmask).buffer(1)
        actor = unary_union(materials + [actual_head]).buffer(8, join_style=1)
        selected = [(n, c, g.intersection(actor)) for n, c, g in paints
                    if c not in BG]
    selected = [(n, c, g) for n, c, g in selected
                if not g.is_empty and sum(p.area for p in polygons(g)) > 0.2]
    headpieces = [g.intersection(headmask) for _, _, g in selected]
    hb = unary_union(headpieces).bounds
    if not hb or len(hb) != 4 or any(not math.isfinite(x) for x in hb):
        raise ValueError(f'bad head box for {index}: {hb}')
    return selected, headmask, hb
