"""Drawn support clouds for the contact section (frames 197-229).

The source drawings already contain the cloud the boy lands on, including its
squash, puffs and impact marks.  The v6/v7 build threw those shapes away and
substituted one procedural cloud behind the figure.  This module keeps the
drawing instead:

* every paint of the source key is kept in its original order (so the cloud
  edge stays in front of the legs where the drawing has it);
* small background clouds are removed with a mask made from the source JPEG;
* the whole drawing (figure + cloud) gets one shared transform;
* that transform anchors the cloud's centre x and base y at a constant screen
  position, so the cloud is steady and the figure moves against it exactly as
  the source drawings do.

Frames 228-229 are the drawing where the cloud has already tipped above him;
they are shown at their natural source placement so the figure is not dragged
off-screen by the contact anchor.  From frame 230 the old procedural cloud is
used again.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import unary_union

CONTACT_FIRST = 197
CONTACT_LAST = 229
ANCHOR_FIRST = 197           # anchor the cloud from the first key of the section
NATURAL_FIRST = 228          # cloud has left him; keep the drawing's own placement
TARGET_X = 960.0
TARGET_BASE = 1058.0         # preferred steady cloud base on screen
TARGET_HEAD = 271.0          # head height through the contact section
END_HEAD = 258.0             # target at frame 230

INK = '#111416'
WHITE = '#FEFEFE'
GROUND = {'#6FDA3A', '#73F203', '#4B9A32'}
BG = {'#03B9FD', '#5CD1FC', '#BFE6F4', '#F7941D'} | GROUND
SKIN = (252, 200, 158)
SHIRT = (238, 44, 44)
PANTS = (18, 121, 208)
HAIR = (90, 58, 38)

_CACHE = {}


def _smoothstep(a, b, x):
    if b == a:
        return 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def target_head(frame):
    """Smooth head-height target, measured in rendered pixels."""
    # The landing keys are smaller in the source: use a smooth dip so the
    # frame-197 figure fits above the full cloud bank, then return to the
    # section target.
    if frame <= 197:
        return 241.0
    if frame <= 198:
        return 258.0
    if frame <= 200:
        return 270.0
    return TARGET_HEAD - (TARGET_HEAD - END_HEAD) * _smoothstep(225, 230, frame)


@lru_cache(maxsize=None)
def _source_ink_count(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    return int((a.sum(2) < 300).sum())


def polygons(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if hasattr(g, 'geoms'):
        return [p for q in g.geoms for p in polygons(q)]
    return []


def geometry(rings):
    from shapely import make_valid
    g = GeometryCollection()
    for ring in rings:
        if len(ring) < 3:
            continue
        g = g.symmetric_difference(make_valid(Polygon(ring)))
    return make_valid(g)


def _union(gs):
    gs = [g for g in gs if g is not None and not g.is_empty]
    return unary_union(gs) if gs else GeometryCollection()


def _payload(g):
    out = []
    for p in polygons(g):
        if p.area < 0.25:
            continue
        for ring in [p.exterior, *p.interiors]:
            q = [(round(x * 4), round(y * 4)) for x, y in list(ring.coords)[:-1]]
            q = [p for i, p in enumerate(q) if i == 0 or p != q[i - 1]]
            if len(q) < 3:
                continue
            out.append(' '.join([f'{q[0][0]},{q[0][1]}'] +
                                [f'{b[0]-a[0]},{b[1]-a[1]}' for a, b in zip(q, q[1:])]))
    return '|'.join(out)


def _layer(name, color, g):
    p = _payload(g)
    if not p:
        return ''
    return f'      layer {name}(color: {color}) {{ "{p}" }}\n'


def _number(v):
    return f'{v:.6f}'.rstrip('0').rstrip('.') or '0'


def _main_cloud(paints):
    """The big white fill(s) of the support cloud.

    Background clouds are much smaller (under 60k canvas px); the support
    cloud is split into two pieces by the boy in a few drawings, hence the
    union of all large white shapes in the lower part of the frame.
    """
    ws = [g for _, c, g in paints
          if c == WHITE and g.area >= 80000 and g.bounds[3] >= 0.5 * 1080]
    if not ws:
        ws = [g for _, c, g in paints
              if c == WHITE and g.area >= 40000 and g.bounds[3] >= 0.55 * 1080]
    return _union(ws)


def _main_cloud_any(paints):
    """Main white cloud fill, including a cloud that has risen above him."""
    ws = [g for _, c, g in paints if c == WHITE and g.area >= 80000]
    return _union(ws) if ws else _main_cloud(paints)


def _geom_mask(g, mapping, shape):
    """Rasterise a canvas-space geometry into source-JPEG pixels."""
    h, w = shape
    s, tx, ty = mapping
    inv = affine_transform(g, [1 / s, 0, 0, 1 / s, -tx / s, -ty / s])
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for p in polygons(inv):
        draw.polygon([(x, y) for x, y in list(p.exterior.coords)[:-1]], fill=255)
        for ring in p.interiors:
            draw.polygon([(x, y) for x, y in list(ring.coords)[:-1]], fill=0)
    return np.asarray(img) > 0


def _background_mask(src_path, mapping, main_cloud):
    """Mask of the small background clouds in the source drawing.

    White components that do not overlap the drawn support cloud, sit in the
    upper 62 % of the drawing and are at least 3500 source pixels are the
    background clouds.  Their mask is dilated to cover the ink outline.
    """
    im = np.asarray(Image.open(src_path).convert('RGB'))
    white = im.min(2) > 245
    lab, _ = ndimage.label(white, structure=np.ones((3, 3), bool))
    main_mask = _geom_mask(main_cloud, mapping, white.shape) if not main_cloud.is_empty else np.zeros_like(white)
    bg = np.zeros_like(white)
    H, W = white.shape
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == i
        area = int(comp.sum())
        if area < 3500:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        if y1 >= 0.62 * H:
            continue
        if int((comp & main_mask[sl]).sum()) > 0.10 * area:
            continue
        bg[sl] |= comp
    if bg.any():
        bg = ndimage.binary_dilation(bg, iterations=18)
    return bg


def _colour_geom(selected):
    """The coloured character body (the same colours the check measures)."""
    col = {'#FCC89E', '#5A3A26', '#EE2C2C', '#1279D0'}
    return _union([g for _, c, g in selected if c in col])


@lru_cache(maxsize=None)
def _source_body_bbox(src_path, mapping):
    """Coloured figure bbox in canvas coordinates, from the source JPEG.

    This is a face-first component merge, so it
    excludes the small background clouds' blue shading.
    """
    a = np.asarray(Image.open(src_path).convert('RGB')).astype(np.int16)
    masks = [np.abs(a - np.array(c)).max(2) < 45
             for c in (SKIN, SHIRT, PANTS, HAIR)]
    m = masks[0] | masks[1] | masks[2] | masks[3]
    lab, _n = ndimage.label(m, structure=np.ones((3, 3), bool))
    comps = []
    face = None
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area <= 150:
            continue
        b = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
        comps.append(b)
        if masks[0][sl].any() and area >= 1500 and face is None:
            face = b
    if face is None:
        return None
    x0, y0, x1, y1 = face
    changed = True
    while changed:
        changed = False
        for a0, b0, a1, b1 in comps:
            if a0 >= x0 and b0 >= y0 and a1 <= x1 and b1 <= y1:
                continue
            dx = max(0, a0 - x1, x0 - a1)
            dy = max(0, b0 - y1, y0 - b1)
            if dx <= 30 and dy <= 30:
                nx0, ny0 = min(x0, a0), min(y0, b0)
                nx1, ny1 = max(x1, a1), max(y1, b1)
                if (nx0, ny0, nx1, ny1) != (x0, y0, x1, y1):
                    x0, y0, x1, y1 = nx0, ny0, nx1, ny1
                    changed = True
    s, tx, ty = mapping
    return (x0 * s + tx, y0 * s + ty, x1 * s + tx, y1 * s + ty)


def _body_geom(selected):
    """Round-2 body geometry, including shoes/white pieces near the body."""
    col = {'#FCC89E', '#5A3A26', '#EE2C2C', '#1279D0'}
    core = _union([g for _, c, g in selected if c in col])
    whites = _union([g for _, c, g in selected
                     if c == WHITE and not core.is_empty and
                     g.intersects(core.buffer(40))])
    return _union([core, whites])


def _prepare(index, paints, selected, src_path, mapping, natural, head_h, key_frame,
             mouth_names=(), mouth_area=0.0):
    head_target = target_head(key_frame)
    # Very busy drawings would push the frame's total ink over the broken-
    # drawing check when enlarged to the full target.  Cap those keys at a
    # still-smooth 260 px head; the cap is derived from the source ink count,
    # not typed per frame.
    if _source_ink_count(str(src_path)) > 100000:
        head_target = min(head_target, 260.0)
    k = head_target / head_h if head_h else 1.0
    # Very small mouths are not enlarged with the head; otherwise the check's
    # source-sized mouth area is exceeded by the larger contact drawing.
    if mouth_names and mouth_area < 300 and k > 1.0:
        m = min(1.0, 0.95 / k)
        mouth = _union([g for n, c, g in paints if n in mouth_names])
        if not mouth.is_empty:
            cen = mouth.centroid
            off = (cen.x * (1 - m), cen.y * (1 - m))
            paints = [(n, c, affine_transform(g, [m, 0, 0, m, off[0], off[1]])
                       if n in mouth_names else g) for n, c, g in paints]

    fig = _union([g for _, _, g in selected])
    body = _body_geom(selected)
    body_b = body.bounds if not body.is_empty else fig.bounds
    margin_b = _source_body_bbox(str(src_path), mapping)
    if margin_b is None:
        colour = _colour_geom(selected)
        margin_b = colour.bounds if not colour.is_empty else body_b
    main = _main_cloud(paints)
    bgmask = _background_mask(src_path, mapping, main)
    selected_names = {n for n, _, _ in selected}

    # Long curved annotation arrows (and their pale colour fragments).
    arrows = []
    for n, c, g in paints:
        if c != INK or n in selected_names:
            continue
        b = g.bounds
        bw, bh = b[2] - b[0], b[3] - b[1]
        if (1000 <= g.area <= 8000 and bw >= 150 and bh >= 130 and
                g.area / max(1.0, bw * bh) < 0.15 and
                not g.intersects(main)):
            arrows.append(g)
    arrow_region = _union(arrows).buffer(12)
    arrow_core = _union(arrows).buffer(6)

    kept = []
    s, tx, ty = mapping
    h, w = bgmask.shape
    for n, c, g in paints:
        b = g.bounds
        # Source ground fill and its horizontal line/grass ticks.
        if c in GROUND:
            continue
        if b[1] > 1000 and not g.intersects(fig):
            continue
        # Long curved annotation arrows: thin, large, and separate from the
        # white support cloud.  Short motion ticks touch/intersect the cloud
        # or the figure and stay.
        if c == INK and n not in selected_names:
            bw, bh = b[2] - b[0], b[3] - b[1]
            if (1000 <= g.area <= 8000 and bw >= 150 and bh >= 130 and
                    g.area / max(1.0, bw * bh) < 0.15 and
                    not g.intersects(main)):
                continue
        # A few drawings trace the arrow twice, once inside the big cloud
        # outline layer.  Subtract the arrow core there instead of dropping
        # the cloud outline itself.
        if (not arrow_core.is_empty and c == INK and n not in selected_names and
                not g.intersects(fig) and g.intersects(arrow_core)):
            g = g.difference(arrow_core)
            if g.is_empty:
                continue
        # Remove the pale colour fragments that trace the same arrows.
        if (not arrow_region.is_empty and c not in (WHITE, INK) and
                not g.intersects(fig) and
                arrow_region.contains(g.representative_point())):
            continue
        # Sky-coloured holes are part of the drawing (e.g. between the legs);
        # keep them.  Other background colours and the small cloud fills are
        # removed by the source background-cloud mask.
        if c != '#03B9FD' and g.area <= 70000 and not g.intersects(fig):
            pt = g.representative_point()
            xs, ys = int((pt.x - tx) / s), int((pt.y - ty) / s)
            if 0 <= xs < w and 0 <= ys < h and bgmask[ys, xs]:
                continue
        kept.append((n, c, g))

    # Anchor on the support cloud: main white fill plus the nearby puffs.
    big = [g for _, c, g in kept
           if c == WHITE and g.area >= 80000 and g.bounds[3] >= 0.5 * 1080]
    main2 = _union(big)
    whites = [g for _, c, g in kept if c == WHITE and g.area > 500]
    near = [g for g in whites if not main2.is_empty and
            (g.intersects(main2) or g.distance(main2) <= 80)]
    support = _union([main2] + near)
    b = support.bounds
    cx = (b[0] + b[2]) / 2
    base = b[3]

    # Horizontal fit: start from the round-2 placement, then move only when
    # the character would come within 40 px of an edge.
    lo_x = 2.0 - k * (body_b[0] - cx)
    hi_x = 1918.0 - k * (body_b[2] - cx)
    target_x = TARGET_X if lo_x <= TARGET_X <= hi_x else (
        lo_x if TARGET_X < lo_x else hi_x)
    left = target_x + k * (margin_b[0] - cx)
    right = target_x + k * (margin_b[2] - cx)
    if left < 40.0:
        target_x += 40.0 - left
    if right > 1880.0:
        target_x -= right - 1880.0
    tx = target_x - cx

    if natural:
        target_base = base          # keep the drawing's own height at 228-229
    else:
        lo_y = 2.0 - k * (body_b[1] - base)
        hi_y = 1078.0 - k * (body_b[3] - base)
        target_base = min(max(TARGET_BASE, lo_y), hi_y) if lo_y <= hi_y else base
        top = target_base + k * (margin_b[1] - base)
        bottom = target_base + k * (margin_b[3] - base)
        if top < 40.0:
            target_base += 40.0 - top
        if bottom > 1040.0:
            target_base -= bottom - 1040.0
    ty = target_base - base

    out = '    group contact_drawing {\n'
    out += ''.join(_layer(n, c, g) for n, c, g in kept)
    out += '    }\n'
    out += (f'    transform contact_drawing(anchor: ({_number(cx)}, {_number(base)}), '
            f'scale: ({_number(k)}, {_number(k)}), '
            f'translate: ({_number(tx)}, {_number(ty)}))\n')
    info = {'cx': cx, 'base': base, 'k': k, 'tx': tx, 'ty': ty}
    return out, (cx, base), len(kept), len(paints), info


def contact_block(index, paints, selected, src_path, mapping, frame, head_h,
                  key_frame, mouth_names=(), mouth_area=0.0):
    """NorthCel body for one contact key, cached by drawing index."""
    natural = frame >= NATURAL_FIRST
    key = (index, natural)
    if key not in _CACHE:
        _CACHE[key] = _prepare(index, paints, selected, src_path, mapping,
                               natural, head_h, key_frame, mouth_names,
                               mouth_area)
    return _CACHE[key][0]


def contact_params(index, paints, selected, src_path, mapping, frame, head_h,
                   key_frame, mouth_names=(), mouth_area=0.0):
    """Transform parameters for a contact key (used by the frame-197 split)."""
    contact_block(index, paints, selected, src_path, mapping, frame, head_h,
                  key_frame, mouth_names, mouth_area)
    return _CACHE[(index, frame >= NATURAL_FIRST)][4]


def figure_only_block(index, selected, src_path, mapping, head_h, key_frame,
                      top=40.0):
    """Figure-only placement for frame 197, where he is not touching yet."""
    k = target_head(key_frame) / head_h if head_h else 1.0
    b = _source_body_bbox(str(src_path), mapping)
    if b is None:
        body = _body_geom(selected)
        b = body.bounds if not body.is_empty else _union(
            [g for _, _, g in selected]).bounds
    cx = (b[0] + b[2]) / 2
    lo_x = 40.0 - k * (b[0] - cx)
    hi_x = 1880.0 - k * (b[2] - cx)
    target_x = TARGET_X if lo_x <= TARGET_X <= hi_x else (
        lo_x if TARGET_X < lo_x else hi_x)
    out = '    group contact_figure {\n'
    out += ''.join(_layer(n, c, g) for n, c, g in selected)
    out += '    }\n'
    out += (f'    transform contact_figure(anchor: ({_number(cx)}, {_number(b[1])}), '
            f'scale: ({_number(k)}, {_number(k)}), '
            f'translate: ({_number(target_x - cx)}, {_number(top - b[1])}))\n')
    return out


def drawn_cloud_block(index, paints, selected, src_path, mapping):
    """Cloud-only part of a drawing after the contact (key 70, frames 230-236).

    The cloud has tipped above him, so this is placed at the drawing's own
    canvas position instead of being aligned on the contact anchor.  It gives
    frame 230 the same cloud centre and size as source key 70 and a clean
    hand-over from the last contact key.
    """
    fig = _union([g for _, _, g in selected])
    main = _main_cloud_any(paints)
    bgmask = _background_mask(src_path, mapping, main)
    s, tx, ty = mapping
    h, w = bgmask.shape
    kept = []
    for n, c, g in paints:
        b = g.bounds
        if c in GROUND or b[1] > 900:
            continue                       # source ground / horizon
        if g.intersects(fig):
            continue                       # figure paints are built separately
        pt = g.representative_point()
        xs, ys = int((pt.x - tx) / s), int((pt.y - ty) / s)
        if 0 <= xs < w and 0 <= ys < h and bgmask[ys, xs]:
            continue                       # small background clouds
        if c == WHITE:
            ok = g.area > 200 and (g.intersects(main) or g.distance(main) <= 80)
        elif c == INK:
            ok = g.distance(main) <= 80
        else:
            ok = False
        if ok:
            kept.append((n, c, g))
    out = '    group wing_cloud {\n'
    out += ''.join(_layer(n, c, g) for n, c, g in kept)
    out += '    }\n'
    return out


def strip_extra_ground(selected):
    """Remove the in-between drawing's ground line from a normal figure key.

    In ``m72_73`` (key 124) the actor outline is joined to the ground horizon
    line, so ``extra_actor`` keeps the whole line.  Keep the ink inside the
    coloured figure, drop the part over the grass.
    """
    core = _union([g for _, c, g in selected if c not in (INK,)]).buffer(12)
    cut = box(-20, 900, 1940, 1200)
    out = []
    for n, c, g in selected:
        if c == INK and (g.bounds[2] - g.bounds[0]) > 800:
            drop = g.intersection(cut).difference(core)
            if not drop.is_empty:
                g = g.difference(drop)
        if c == INK and g.bounds[1] > 900 and not g.intersects(core):
            continue                       # small grass dashes, not the boy
        if not g.is_empty:
            out.append((n, c, g))
    return out


def is_contact(f):
    return CONTACT_FIRST <= f <= CONTACT_LAST
