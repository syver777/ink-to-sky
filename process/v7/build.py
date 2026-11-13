#!/usr/bin/env python3
"""Ink to Sky build.

Builds the 300-frame film from the NorthCel drawings in this repository. It
reads only files here:

  process/v6/authoring.json          key slots and cel timing
  process/polish/extra_keys.json     AI in-between timings
  ncel/sources/cels/*.ncel           traced 1536x1024 source cels
  ncel/sources/extras/*.ncel         traced AI in-betweens
  ncel/sources/receipts/*.json       fit receipts (shape boxes)
  ncel/authored/*.ncel               read-only ground truth for extras

and writes ncel/v7/film.ncel plus, unless --no-render is given, frames and an
mp4 under --out.

Run:
    python process/v7/build.py --out out
    python process/v7/build.py --out out --frames 190..230
    python process/v7/build.py --no-render
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import northcel
from northcel import renderer
from shapely import make_valid
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import v7lib as L  # noqa: E402
import v7parts as P  # noqa: E402
import cloud as C  # noqa: E402
import face as F  # noqa: E402

W, H = 1920, 1080
INK = '#111416'
WHITE = '#FEFEFE'
SKY = '#03B9FD'
GROUND = '#6FDA3A'  # hill green (palette c2)
GRASS = '#297427'   # grass-blade green
HEADER = ('northcel "0.6.0-draft"\nfilm InkToSkyV7(width: 1920, height: 1080, fps: 20, '
          'frames: 300, precision: 0.25px, seed: 42) {\n  paper { color: #FEFEFE, antialias_grid: 4 }\n')

# Key slots whose source drawing is NOT used as a key pose.
UNUSED_EXTRA = {10, 23, 54, 84}  # ids 101, 103, 114, 125 (reviewer-rejected)
USE_PARTS5 = {59, 65, 67, 69, 71}  # cloud-pose keys the v4 actor rule cannot split

# Camera/acting landmarks (v6 paths, unchanged).
X = [(0, 960), (7, 939), (10, 919), (13, 898), (17, 880), (20, 914), (27, 956), (30, 980), (33, 1005), (37, 985),
     (50, 960), (100, 960), (103, 920), (107, 860), (110, 960), (120, 960), (127, 960), (180, 960), (193, 970),
     (200, 960), (203, 960), (210, 960), (217, 960), (220, 1000), (227, 1170), (230, 1120), (237, 950), (240, 950)]
Y = [(0, 515), (37, 515), (50, 220), (100, 220), (103, 370), (107, 470), (110, 220), (117, 200), (120, 180),
     (127, 300), (140, 350), (180, 380), (193, 285), (197, 260), (200, 350), (203, 510), (207, 590), (210, 385),
     (213, 550), (217, 535), (220, 520), (223, 530), (227, 560), (230, 450), (237, 450), (240, 406)]
S = [(0, 455), (37, 455), (50, 310), (100, 310), (107, 310), (120, 310), (127, 350), (193, 350), (197, 305), (200, 305), (203, 350), (240, 300)]
ANGLE = [(0, 0), (203, 0), (217, 0), (220, 5), (223, 10), (227, 19), (230, 12), (237, 0), (240, 0)]

CLOUD_LOBES = [
    (-330, -58, 92, 84), (-190, -96, 96, 92), (-40, -118, 100, 98),
    (110, -100, 94, 90), (255, -70, 88, 80), (365, -38, 72, 62),
    (-380, -30, 64, 58), (-250, -30, 120, 62), (-80, -26, 150, 58),
    (110, -26, 140, 56), (270, -26, 110, 52), (380, -18, 60, 44),
]


# ---------------------------------------------------------------------------
# basic geometry helpers (v6 verbatim)
# ---------------------------------------------------------------------------
def polygons(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if hasattr(g, 'geoms'):
        return [p for q in g.geoms for p in polygons(q)]
    return []


def geometry(rings):
    g = GeometryCollection()
    for ring in rings:
        if len(ring) < 3:
            continue
        p = make_valid(Polygon(ring))
        g = g.symmetric_difference(p)
    return make_valid(g)


def payload(g):
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


def layer(name, color, g, outline=False, outline_color=INK):
    p = payload(g)
    if not p:
        return ''
    kind = 'cel' if outline else 'layer'
    extra = f', outline: ({outline_color}, 4.5px)' if outline else ''
    return f'      {kind} {name}(color: {color}{extra}) {{ "{p}" }}\n'


def number(v):
    return f'{v:.6f}'.rstrip('0').rstrip('.') or '0'


def smooth(a, b):
    if b == a:
        return '0'
    u = f'((f - {a}) / {b-a})'
    return f'(0.5 * {u} + 0.5 * smoothstep({a}, {b}, f))'


def expr_path_sym(points):
    out = number(points[0][1])
    for (a, x), (b, y) in zip(points, points[1:]):
        out = f'({out} + {number(y - x)} * smoothstep({a}, {b}, f))'
    return out


def expr_path(points, f):
    if f <= points[0][0]:
        return number(points[0][1])
    for (a, x), (b, y) in zip(points, points[1:]):
        if f < b:
            return f'({number(x)} + {number(y-x)} * {smooth(a,b)})'
    return number(points[-1][1])


# ---------------------------------------------------------------------------
# inventory of source keys
# ---------------------------------------------------------------------------
def _cel_path(key):
    kid = key['id']
    if kid < 100:
        return ROOT / 'ncel/sources/cels' / f'f{kid:04d}.ncel'
    src = key['_extra_src']
    return ROOT / 'ncel/sources/extras' / (Path(src).stem + '.ncel')


def _receipt(key):
    path = ROOT / 'ncel/sources/receipts' / (_cel_path(key).stem + '.json')
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _source_path(index):
    """The committed source JPEG for a key (used only for cloud masking)."""
    key = INV[index]
    if key['id'] < 100:
        return ROOT / 'sources/cels' / f"f{key['id']:04d}.jpg"
    return ROOT / 'sources/extras' / (Path(key['_extra_src']).stem + '.jpg')


def _class_from(text, receipt, paints):
    paper = _paper_colour(text)
    if paper and paper != WHITE:
        return 'full'
    if receipt:
        bb = [1e9, 1e9, -1e9, -1e9]
        for shape in receipt['shape_manifest']:
            b = shape['bbox']
            bb = [min(bb[0], b[0]), min(bb[1], b[1]),
                  max(bb[2], b[2]), max(bb[3], b[3])]
        if bb[0] < 5 and bb[2] > 1531:
            return 'full'
    g = unary_union([geometry(p.rings) for p in paints])
    b = g.bounds
    return 'full' if (b[0] < 5 and b[2] > 1531) else 'paper'


def _paper_colour(text):
    m = re.search(r'paper \{ color: (\$c\d+|#[0-9A-Fa-f]{6})', text)
    if not m:
        return None
    value = m.group(1)
    if value.startswith('$'):
        pal = dict(re.findall(r'(c\d+): (#[0-9A-Fa-f]{6})', text))
        value = pal.get(value[1:], '#FEFEFE')
    return value.upper()


@lru_cache(maxsize=None)
def _key_text(index):
    return _cel_path(INV[index]).read_text()


@lru_cache(maxsize=None)
def _key_paints(index):
    d = northcel.parse_ncel(_key_text(index))
    return renderer.frame_paint_list(d, 0)[2]


def _key_class(index):
    """'paper' / 'full' scene mapping for a source key.

    Paper scenes fit whole in the 1920x1080 canvas (scale 1.0546875, x +150).
    Full-bleed scenes cover it (scale 1.25, y -100).  Extras are 1536x1024
    drawings too, and use the same two mappings in v7.
    """
    key = INV[index]
    return _class_from(_key_text(index), _receipt(key), _key_paints(index))


def _mapping(index):
    if INV[index]['id'] >= 100:
        # Extras are drawn on the same 1536x1024 sheet as the cels.  The
        # reference film shows them at natural size in the canvas (contain,
        # no upscaling), not at the cels' fit-whole scale.
        return (1.25, 0.0, -100.0) if _key_class(index) == 'full' else (1.01, 184.5, 23.0)
    return (1.25, 0.0, -100.0) if _key_class(index) == 'full' else (1.0546875, 150.0, 0.0)


@lru_cache(maxsize=None)
def raw(index):
    """Reconstructed reference key: (name, colour, geometry) on the canvas."""
    s, tx, ty = _mapping(index)
    out = []
    for n, p in enumerate(_key_paints(index)):
        g = affine_transform(geometry(p.rings), [s, 0, 0, s, tx, ty])
        if g.is_empty:
            continue
        out.append((f'k{index}_{n}', p.color.to_hex(), g))
    return out


def inventory():
    """Full 86-key ordered inventory (61 used cels + 21 used extras + 4 rejected)."""
    auth = json.loads((ROOT / 'process/v6/authoring.json').read_text())
    extras = json.loads((ROOT / 'process/polish/extra_keys.json').read_text())
    extra_by_id = {e['id']: e for e in extras}
    slots = auth['key_slots']
    by_id = {s['key_id']: s for s in slots}

    rows = []
    for s in slots:
        row = dict(index=s['source'], id=s['key_id'], frame=s['frame'],
                   used=True, extra=bool(s['extra']))
        if s['extra']:
            row['_extra_src'] = extra_by_id[s['key_id']]['src']
        rows.append(row)

    # The four rejected extras keep their place in the inventory (v6 numbered
    # all 86 keys) but are never used as key poses.
    for index, key_id in {10: 101, 23: 103, 54: 114, 84: 125}.items():
        e = extra_by_id[key_id]
        fa = by_id[e['a']]['frame']
        fb = by_id[e['b']]['frame']
        rows.append(dict(index=index, id=key_id, frame=fa + e['u'] * (fb - fa),
                         used=False, extra=True, _extra_src=e['src']))
    rows.sort(key=lambda k: (k['frame'], k['index']))
    for i, row in enumerate(rows):
        row['index'] = i
    assert len(rows) == 86

    # Shape boxes from the committed fit receipts, mapped to the canvas.
    for row in rows:
        receipt = _receipt(row)
        if not receipt:
            continue
        text = _cel_path(row).read_text()
        d = northcel.parse_ncel(text)
        paints = renderer.frame_paint_list(d, 0)[2]
        kind = _class_from(text, receipt, paints)
        s, tx, ty = (1.25, 0.0, -100.0) if kind == 'full' else (1.0546875, 150.0, 0.0)
        shapes = []
        for n, shape in enumerate(receipt['shape_manifest']):
            b = shape['bbox']
            shapes.append(dict(id=f"k{row['index']}_{n}",
                               bbox=[b[0] * s + tx, b[1] * s + ty,
                                     b[2] * s + tx, b[3] * s + ty],
                               color='#%02X%02X%02X' % tuple(shape['color'])))
        row['shapes'] = shapes
    return rows


INV = inventory()
USABLE = [k for k in INV if k['used']]
_KEY_INDEX = {k['index']: k for k in INV}


def _authored_anchor(index):
    """Centre of the authored raw block that copies this source key.

    `ncel/authored/*.ncel` are ground truth for the reference keys.  Their
    raw layer union tells us where the figure sat in the 1536x1024 drawing;
    extras are rendered at that natural position instead of being normalised
    through a head box.
    """
    for fname in sorted(glob.glob(str(ROOT / 'ncel/authored/*.ncel'))):
        text = Path(fname).read_text()
        m = re.search(r'Copy source key (\d+)', text)
        if not m or int(m.group(1)) != index:
            continue
        geoms = []
        for lm in re.finditer(
                r'layer ([\w]+)\(color: (#[0-9A-Fa-f]{6})\) \{ "(.*?)" \}', text, re.S):
            name = lm.group(1)
            if name.startswith(('bg_cloud', 'ground', 'grass_', 'support_',
                                'crease_', 'duck_', 'pen')):
                continue
            geoms.append(geometry(northcel.decode_rings_payload(lm.group(3), '0.25px')))
        if not geoms:
            return None
        u = unary_union(geoms).bounds
        return Polygon([(u[0], u[1]), (u[2], u[1]), (u[2], u[3]), (u[0], u[3])]).centroid
    return None


@lru_cache(maxsize=None)
def extra_actor(index):
    """Cut the actor out of a native extra drawing, in canvas coordinates.

    The reference keys for extras were fitted on the 1536x1024 drawing
    itself.  Their actor is the ink contour that contains the authored anchor:
    clouds and motion lines are separate components.  Fill that contour, then
    keep the character shapes inside it.
    """
    paints = raw(index)
    if _key_class(index) == 'paper':
        return [x for x in paints if x[1] not in P.BG]
    anchor = _authored_anchor(index)
    s, tx, ty = _mapping(index)

    ink_polys = [p for _, c, g in paints if c == INK for p in polygons(g)]
    chosen = None
    if anchor is not None:
        pt = affine_transform(anchor, [s, 0, 0, s, tx, ty])
        containing = [p for p in ink_polys if p.buffer(2).contains(pt)]
        if containing:
            # the figure contour is the largest contour containing the anchor
            # (cloud contours are separate and smaller)
            chosen = max(containing, key=lambda p: p.area)
        else:
            chosen = min(ink_polys, key=lambda p: p.distance(pt) - p.area * 1e-6,
                         default=None)
    if chosen is None:
        chosen = max(ink_polys, key=lambda p: p.area, default=None)
    if chosen is None:
        return [x for x in paints if x[1] not in P.BG]

    filled = Polygon(chosen.exterior)
    for p in ink_polys:
        if p is chosen:
            continue
        if filled.buffer(1).contains(p.representative_point()) or \
                p.intersection(filled).area > 0.5 * p.area:
            filled = filled.union(Polygon(p.exterior))

    selected = []
    for n, c, g in paints:
        if c in P.BG:
            continue
        if filled.buffer(1).contains(g.representative_point()) or \
                g.intersection(filled).area > 0.5 * max(1.0, g.area):
            if sum(p.area for p in polygons(g)) > 0.2:
                selected.append((n, c, g))

    # A few AI drawings contain the cloud/impact puffs as part of the same
    # connected ink contour as the boy.  Use the coloured character core as a
    # region of interest so those props do not join the figure.
    huge_white = any(c == WHITE and g.area > 100000 for _, c, g in selected)
    core = [g for _, c, g in selected
            if c in ('#FCC89E', '#5A3A26', '#EE2C2C', '#1279D0') and g.area > 500]
    if core and huge_white:
        b = unary_union(core).bounds
        w, h = b[2] - b[0], b[3] - b[1]
        roi = box(b[0] - 0.35 * w - 20, b[1] - 0.35 * h - 20,
                  b[2] + 0.35 * w + 20, b[3] + 0.30 * h + 20)
        pruned = []
        for n, c, g in selected:
            if c == WHITE and g.area > 100000:
                continue  # drawn cloud fill, not the boy
            rep = g.representative_point()
            if roi.contains(rep):
                if g.area > 20000 and not roi.contains(g):
                    q = g.intersection(roi)
                    if not q.is_empty and sum(p.area for p in polygons(q)) > 0.2:
                        pruned.append((n, c, q))
                else:
                    pruned.append((n, c, g))
            elif g.area < 12000 and g.intersection(roi).area > 0.6 * g.area:
                pruned.append((n, c, g))
        if pruned:
            selected = pruned

    return selected


@lru_cache(maxsize=None)
def _plan_head(index):
    """Head box stored in the committed plan's figure transform (native coords).

    The reference keys for AI extras are not mapped to the canvas; the
    committed plan recorded their transform anchor and head height.  We use
    those numbers (they are repository ground truth) instead of asking a
    fragile face detection to re-derive them from a re-fitted extra.
    """
    j = USABLE.index(INV[index])
    text = (ROOT / 'ncel/film/film.ncel').read_text()
    m = re.search(r'^  def K%d \{\n' % j, text, re.M)
    if not m:
        return None
    seg = text[m.end():]
    nxt = re.search(r'^  def K\d+ \{\n', seg, re.M)
    if nxt:
        seg = seg[:nxt.start()]
    pos = seg.find('transform figure(')
    if pos < 0:
        return None
    seg = seg[pos:]
    am = re.search(r'anchor: \(([-\d.]+), ([-\d.]+)\)', seg)
    if not am:
        return None
    hx, hy = float(am.group(1)), float(am.group(2))
    sm = re.search(r'scale: \(', seg)
    if not sm:
        return None
    depth = 0
    start = sm.end()
    for i in range(start, len(seg)):
        ch = seg[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            if depth == 0:
                break
            depth -= 1
        elif ch == ',' and depth == 0:
            s1 = seg[start:i]
            den = s1.rsplit('/', 1)[-1].strip()
            try:
                hh = float(den)
            except ValueError:
                return None
            return hx, hy, hh
    return None


@lru_cache(maxsize=None)
def _authored_head(index):
    """Head box stored in the authored raw block that copies this source key.

    The authored blocks are repository ground truth for the reference
    keys.  Their figure transform carries the same anchor/head-height numbers
    the build uses, in the drawing's native coordinates.
    """
    for fname in sorted(glob.glob(str(ROOT / 'ncel/authored/*.ncel'))):
        text = Path(fname).read_text()
        m = re.search(r'Copy source key (\d+)', text)
        if not m or int(m.group(1)) != index:
            continue
        pos = text.find('transform figure(')
        if pos < 0:
            return None
        seg = text[pos:]
        am = re.search(r'anchor: \(([-\d.]+), ([-\d.]+)\)', seg)
        if not am:
            return None
        hx, hy = float(am.group(1)), float(am.group(2))
        sm = re.search(r'scale: \(', seg)
        if not sm:
            return None
        depth = 0
        start = sm.end()
        for i in range(start, len(seg)):
            ch = seg[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth == 0:
                    break
                depth -= 1
            elif ch == ',' and depth == 0:
                den = seg[start:i].rsplit('/', 1)[-1].strip()
                try:
                    return hx, hy, float(den)
                except ValueError:
                    return None
    return None


@lru_cache(maxsize=None)
def extra_parts(index):
    """Head mask and head box for an extra actor."""
    sel = extra_actor(index)
    ph = None
    if _paper_colour(_key_text(index)) == WHITE:
        ph = _authored_head(index) or _plan_head(index)
    if ph is not None:
        hx, hy, hh = ph
        s, tx, ty = _mapping(index)
        hx, hy, hh = hx * s + tx, hy * s + ty, hh * s
        hb = (hx - hh / 2, hy - hh / 2, hx + hh / 2, hy + hh / 2)
        headmask = box(hb[0] - 42, hb[1] - 72, hb[2] + 42, hb[3] + 8)
        return sel, headmask, hb
    fb = P.face_box(sel)
    headmask = box(fb[0] - 42, fb[1] - 72, fb[2] + 42, fb[3] + 8)
    hp = [g.intersection(headmask) for _, _, g in sel]
    hp = [g for g in hp if not g.is_empty]
    hb = unary_union(hp).bounds
    return sel, headmask, hb


@lru_cache(maxsize=None)
def parts(index):
    if INV[index]['id'] >= 100:
        return extra_parts(index)
    if index in USE_PARTS5:
        # v6parts.parts5 first cuts the actor, then finds the face fill.
        actor, _ = P.actor_paints(raw(index))
        return P.parts5(actor, fills_only=False)
    return P.parts_v4(raw(index), index)


# ---------------------------------------------------------------------------
# film content (v6 port)
# ---------------------------------------------------------------------------
def source_hill(base=740.0):
    # the hill is measured from the key-76 raster
    # from raw(76), so rasterise that key at canvas size.
    return L.source_hill(raw(76), base=base)


HILL = source_hill(740.0)


def background(f):
    if f < 123:
        return ''
    out = f'    paper {{ color: mix({WHITE}, {SKY}, smoothstep(123, 140, f)), antialias_grid: 4 }}\n'
    out += '    group background_clouds {\n'
    for j, (x, y, w, h) in enumerate([(260, 160, 290, 130), (1580, 170, 250, 110),
                                      (180, 590, 190, 90), (1630, 570, 220, 100),
                                      (550, 850, 140, 65)]):
        lobes = []
        for dx, dy, rx, ry in [(-.28, .12, .25, .29), (-.08, -.05, .27, .43), (.20, .04, .32, .35)]:
            ring = [(x + w * (dx + rx * math.cos(a)), y + h * (dy + ry * math.sin(a)))
                    for a in np.linspace(0, math.tau, 40, endpoint=False)]
            lobes.append(Polygon(ring))
        out += f'      group bg_cloud_{j}(pivot: ({x+w/2:.1f}, {y+h/2:.1f}), opacity: smoothstep({124+j*2}, {134+j*2}, f)) {{\n'
        edge = '#1279D0' if C.is_contact(f) else INK
        out += layer(f'bg_cloud_{j}_fill', WHITE, unary_union(lobes), True, edge)
        out += '      }\n'
    out += '    }\n'
    for j in range(5):
        out += (f'    transform bg_cloud_{j}(scale: (0.86 + 0.14 * smoothstep({124+j*2}, {134+j*2}, f), '
                f'0.86 + 0.14 * smoothstep({124+j*2}, {134+j*2}, f)), '
                f'translate: (0, 26 * (1 - smoothstep({124+j*2}, {134+j*2}, f))))\n')
    out += '    transform background_clouds(translate: (0, -0.22 * (f - 130)))\n'
    if f >= 230:
        top = list(HILL)
        top = [(-40, top[0][1])] + top + [(1960, top[-1][1])]
        out += layer('ground', GROUND, Polygon(top + [(1960, 1300), (-40, 1300)]), True)
        out += '    transform ground(translate: (0, 360 * (1 - smoothstep(230, 241, f))))\n'
        if not 238 <= f <= 239:
            out += '    group grass_marks {\n'
            for j in range(26):
                x = 50 + (j * 151) % 1820
                y = 790 + (j * 83) % 250
                out += f'      stroke grass_{j}(color: {GRASS}, width: 2px) {{ ({x-5}, {y-7}), ({x}, {y}), ({x+6}, {y-10}) }}\n'
            out += '    }\n    transform grass_marks(translate: (0, 360 * (1 - smoothstep(230, 241, f))))\n'
    return out


def cloud_shape():
    lobes = []
    for dx, dy, rx, ry in CLOUD_LOBES:
        lobes.append(Polygon([(dx + rx * math.cos(a), dy + ry * math.sin(a))
                              for a in np.linspace(0, math.tau, 56, endpoint=False)]))
    return unary_union(lobes)


CLOUD = cloud_shape()


def support(f):
    """Fluffy cloud: under him, then tipped above as he falls (key 69/70)."""
    if not 197 <= f < 245 or C.is_contact(f) or 237 <= f <= 241:
        return ''
    PX = [(197, 1000), (217, 1000), (220, 1070), (227, 1170), (230, 1000), (235, 920), (240, 840), (244, 760)]
    PY = [(197, 905), (217, 905), (223, 900), (227, 860), (228, 720), (229, 540), (230, 360),
          (234, 290), (236, 120), (238, -160), (244, -600)]
    pxexpr = expr_path(PX, f)
    if f < 230:
        py = expr_path(PY, f)
        rot = expr_path([(197, 0), (217, 0), (220, 3), (223, 9), (227, 20), (229, 26)], f)
        sq = '(1 - 0.10 * smoothstep(203, 214, f) - 0.16 * smoothstep(217, 229, f))'
    else:
        py = expr_path(PY, f)
        rot = expr_path([(230, 28), (235, 34), (240, 40), (244, 46)], f)
        sq = '1'
    out = '    group support_cloud(pivot: (0, 0)) {\n'
    out += layer('support_fill', WHITE, CLOUD, True)
    for i, (x, y) in enumerate([(-300, -40), (-140, -70), (20, -72), (185, -46)]):
        out += f'      stroke crease_{i}(color: {INK}, width: 3px) {{ ({x-58}, {y+10}), ({x-20}, {y-5}), ({x+20}, {y-6}), ({x+52}, {y+6}) }}\n'
    out += '    }\n'
    out += f'    transform support_cloud(translate: ({pxexpr}, {py}), rotate: {rot})\n'
    out += f'    transform support_cloud(scale: (1, {sq}))\n'
    if f >= 236:
        out += f'    transform support_cloud(translate: ({-28 * (f - 236)}, {-9 * (f - 236)}))\n'
    return out


@lru_cache(maxsize=None)
def duck_geometry():
    paints = raw(79)
    candidates = [g for _, c, g in paints if c == INK and not g.is_empty and
                  g.bounds[0] > 290 and g.bounds[2] < 550 and
                  g.bounds[1] > 870 and g.area > 5000]
    main = max(candidates, key=lambda g: g.area)
    mask = main.buffer(1)
    return [(n, c, g.intersection(mask)) for n, c, g in paints
            if c not in C.GROUND and c != SKY
            and not g.intersection(mask).is_empty], main.bounds


def duck(f):
    if f < 243:
        return ''
    paints, bounds = duck_geometry()
    cx = (bounds[0] + bounds[2]) / 2
    by = bounds[3]
    out = f'    group duck(pivot: ({cx}, {by})) {{\n'
    for n, c, g in paints:
        out += layer('duck_' + n, c, g)
    out += '    }\n'
    from northcel.rings import decode_rings_payload
    # The right foot is the rightmost orange shape in the lower part of the
    # drawing.  Shape numbers move when a drawing is re-fitted, so locate it
    # by colour and position instead of by name.
    feet = [q for q in paints
            if q[1] == '#F7941D' and q[2].bounds[1] > 1000]
    foot = max(feet, key=lambda q: (q[2].bounds[2], q[2].area), default=None)
    if foot:
        rings = decode_rings_payload(payload(foot[2]), '0.25px')
        vertex = max(range(len(rings[0])), key=lambda i: rings[0][i][0])
        out += f'    nudge duck_{foot[0]}(vertex: {vertex}, ring: 0, dx: 2 * sin(f * 1.1), dy: 0)\n'
    out += f'    warp duck(handles: (({cx}, {by}, 0, 0), ({cx}, {by-150}, 3 * sin(f * 1.1), 0)), radius: 220)\n'
    out += (f'    transform duck(scale: (1.17, 1.17), rotate: 2.5 * sin(f * 1.1), '
            f'translate: ({620 - cx + 25 * (f - 243)}, {990 - by - 2.5 * (1 - math.cos(f * 2.2))}))\n')
    return out


@lru_cache(maxsize=None)
def pen_delta(a, b):
    _, _, hb = parts(a)

    def visible_ink(index):
        result = GeometryCollection()
        for _, c, g in raw(index):
            result = result.union(g) if c == INK else result.difference(g)
        return make_valid(result)

    ga = visible_ink(a)
    gb = visible_ink(b)
    return gb.difference(ga.buffer(1.5)).intersection(box(300, hb[3] - 2, 1600, 1200))


def usable_keys():
    return [k for i, k in enumerate(INV) if k['id'] < 100 or i not in UNUSED_EXTRA]


def source_at(f):
    a = max((k for k in usable_keys() if k['frame'] <= f), key=lambda k: k['frame'])
    return a['index']


def figure_block(index, f, symbolic=False):
    """v6 figure_block.  Cels only; extras are placed at natural size."""
    selected, mask, hb = parts(index)
    if INV[index]['id'] == 124:
        selected = C.strip_extra_ground(selected)
    selected = F.face_shapes(index, selected)
    hx = (hb[0] + hb[2]) / 2
    hy = (hb[1] + hb[3]) / 2
    hh = hb[3] - hb[1]
    out = '    group figure {\n' + ''.join(layer(n, c, g) for n, c, g in selected) + '    }\n'
    if f >= 100:
        out += (f'    warp figure(handles: (({hx}, {hy}, 0, 0), ({hx}, {hy+420}, 5 * sin(f * 0.24), 7 * sin(f * 0.24)), '
                f'({hx-480}, {hy+140}, -5 * sin(f * 0.24), 0), ({hx+480}, {hy+140}, 5 * sin(f * 0.24), 0)), radius: 600)\n')
    float_in = 'smoothstep(40, 52, f) * (1 - smoothstep(92, 100, f))'
    bx = f'(3.0 * sin(f * 0.09) + 1.5 * sin(f * 0.21) + 14 * sin((f - 40) * 0.105) * {float_in})'
    by = f'(2.0 * sin(f * 0.07 + 1.1) + 1.0 * sin(f * 0.17) + 12 * sin((f - 40) * 0.083 + 0.7) * {float_in})'
    path = expr_path_sym if symbolic else (lambda pts: expr_path(pts, f))
    tx = f'({path(X)} + {bx} + 2.0 * sin(f * 0.13))'
    ty = f'({path(Y)} + {by} + 1.5 * sin(f * 0.17 + 0.5))'
    sz = f'({path(S)} * (1 + 0.009 * sin(f * 0.33)))'
    ang = f'({path(ANGLE)} + 0.55 * sin(f * 0.27))'
    out += f'    transform figure(anchor: ({hx}, {hy}), scale: ({sz} / {hh}, {sz} / {hh}), rotate: {ang}, translate: ({tx} - {hx}, {ty} - {hy}))\n'
    out += F.extra_transform(index, f)
    if (index == 52 and f >= 192) or (index == 53 and f >= 193):
        # Smooth the pre-contact head down toward the smaller source pose so
        # frame 197 can sit above the full cloud bank with a 40 px top margin.
        target = number(278) if index == 52 else path([(193, 259), (196, 248)])
        hs = F._source_head(index)[0]
        hs = hs[3] - hs[1]
        fy = max(g.bounds[3] for _, _, g in selected)
        kf = f'({target} / ({number(hs)} * ({sz} / {number(hh)})))'
        out += (f'    transform figure(anchor: ({hx}, {number(fy)}), '
                f'scale: ({kf}, {kf}), translate: (0, 0))\n')
    return out


def contents(f, symbolic=False):
    index = source_at(f)
    a = max((k for k in INV if k['frame'] <= f), key=lambda k: k['frame'])
    b = next((k for k in INV if k['frame'] > f), a)
    if f == 197:
        # He is not touching the cloud yet.  Use the smaller source-60 figure,
        # and borrow the full cloud bank from the next key so 197-199 match.
        selected, _, _ = parts(index)
        hb, _ = F._source_head(index)
        head_h = hb[3] - hb[1] if hb is not None else 1.0
        out = background(f)
        out += C.figure_only_block(index, F.face_shapes(index, selected),
                                   _source_path(index), _mapping(index),
                                   head_h, INV[index]['frame'], top=40.0)
        ci = source_at(198)
        csel, _, _ = parts(ci)
        chb, _ = F._source_head(ci)
        ch = chb[3] - chb[1] if chb is not None else 1.0
        cp = C.contact_params(ci, F.face_shapes(ci, list(raw(ci))),
                              F.face_shapes(ci, csel), _source_path(ci),
                              _mapping(ci), 198, ch, INV[ci]['frame'])
        out += C.drawn_cloud_block(ci, raw(ci), csel,
                                   _source_path(ci), _mapping(ci))
        out += (f'    transform wing_cloud(anchor: ({cp["cx"]}, {cp["base"]}), '
                f'scale: ({cp["k"]}, {cp["k"]}), '
                f'translate: ({cp["tx"]}, {cp["ty"]}))\n')
        out += duck(f)
        return out
    if C.is_contact(f):
        selected, _, _ = parts(index)
        hb, _ = F._source_head(index)
        head_h = hb[3] - hb[1] if hb is not None else 1.0
        mouth = F.mouth_info(index)
        mouth_names = {r['name'] for r in mouth['fills']}
        mouth_area = sum(r['area'] for r in mouth['fills'])
        out = background(f)
        # The contact drawing is built from the whole source key, so the
        # face fixes are applied to those paints as well as to the figure.
        out += C.contact_block(index, F.face_shapes(index, list(raw(index))),
                               F.face_shapes(index, selected),
                               _source_path(index), _mapping(index), f,
                               head_h, INV[index]['frame'],
                               mouth_names, mouth_area)
        out += duck(f)
        return out
    if 230 <= f <= 236:
        selected, _, _ = parts(index)
        dy = -_mapping(index)[2]
        out = background(f)
        out += C.drawn_cloud_block(index, raw(index), selected,
                                   _source_path(index), _mapping(index))
        out += f'    transform wing_cloud(translate: (0, {number(dy)}))\n'
        out += figure_block(index, f, symbolic)
        out += f'    transform figure(translate: (0, {number(dy)}))\n'
        out += duck(f)
        return out
    out = background(f) + support(f)
    if index >= 0 and INV[index]['id'] >= 100:
        # AI in-between key.  Full-bleed extras keep the v6 head normalisation
        # (so they sit on the acting path); paper extras are shown at the
        # drawing's natural size, which is what the committed mp4 does.
        out += figure_block(index, f, symbolic)
        out += duck(f)
        return out
    out += figure_block(index, f, symbolic)
    if 50 <= f < 90 and b['index'] != a['index'] and b['id'] < 100:
        delta = pen_delta(a['index'], b['index'])
        if not delta.is_empty:
            x0, y0, x1, y1 = delta.bounds
            out += '    layer pen_window(tone: 1, opacity: 0) { "1200,0 5200,0 0,4800 -5200,0" }\n'
            u = smooth(a['frame'], b['frame'])
            out += f'    transform pen_window(scale: (1, ({y0} + {y1-y0+2} * {u}) / 1200))\n'
            out += '    group new_pen_marks(parent: figure, clip: pen_window) {\n' + layer(f'pen_{a["id"]}_{b["id"]}', INK, delta) + '    }\n'
    out += duck(f)
    return out


# ---------------------------------------------------------------------------
# build the one complete film
# ---------------------------------------------------------------------------
def build_film():
    F.configure(sys.modules[__name__])
    F.validate_mouths()
    usable = usable_keys()
    breakpoints = {0, 123, 125, 127, 130, 140, 197, 200, 203, 217, 220, 223, 227, 230, 237, 240, 243, 290, 300}
    for seq in [usable, X, Y, S, ANGLE]:
        breakpoints.update(k['frame'] if isinstance(k, dict) else k[0] for k in seq)
    breakpoints = sorted(breakpoints)
    keyframes = {k['frame'] for k in usable}

    defs = []
    exact = []
    for j, k in enumerate(usable):
        f = k['frame']
        body = contents(f).replace('    def KeyA {\n', '').replace('    }\n    use KeyA\n', '')
        body = re.sub(r'\bf\b', str(f), body)
        papers = re.findall(r'^    paper .*$', body, re.M)
        body = re.sub(r'^    paper .*\n', '', body, flags=re.M)
        defs.append(f'  def K{j} {{\n{body}  }}\n')
        exact.append(f'  frame {f} {{\n' + ''.join(p + '\n' for p in papers) + f'    use K{j}\n  }}\n')

    ranges = []
    for lo, hi in zip(breakpoints, breakpoints[1:]):
        remaining = [f for f in range(lo, hi) if f not in keyframes]
        runs = []
        for f in remaining:
            if runs and f == runs[-1][-1] + 1:
                runs[-1].append(f)
            else:
                runs.append([f])
        for seq in runs:
            first, last = seq[0], seq[-1]
            span = str(first) if first == last else f'{first}..{last}'
            ranges.append(f'  frame {span} {{\n' + contents(first, symbolic=True) + '  }\n')
    last = usable[-1]['frame']
    if last < 299 and breakpoints[-1] <= last:
        # normally the 290..300 breakpoint pair already covers the tail
        ranges.append(f'  frame {last+1}..299 {{\n' + contents(last + 1, symbolic=True) + '  }\n')
    return HEADER + ''.join(defs + ranges + exact) + '}\n'


def write_film(path):
    text = build_film()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _parse_frames(spec):
    if not spec:
        return None
    m = re.fullmatch(r'(\d+)(?:\.\.(\d+))?', spec.strip())
    if not m:
        raise SystemExit(f'bad --frames {spec!r}; expected A or A..B')
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) is not None else a
    if not (0 <= a <= b <= 299):
        raise SystemExit(f'--frames out of range: {spec}')
    return list(range(a, b + 1))


def _render_worker(args):
    film_path, out_dir, indices = args
    doc = northcel.parse_ncel(Path(film_path).read_text())
    out = Path(out_dir)
    for idx in indices:
        rgb = northcel.render_document_frame(doc, frame_index=idx)
        from PIL import Image
        Image.fromarray(rgb, mode='RGB').save(out / f'f{idx:04d}.png')
    return len(indices)


def render_frames(film_path, out_dir, indices, workers=6):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not indices:
        return
    # 6 workers by default; each worker parses once and renders its chunk.
    workers = max(1, min(workers, len(indices)))
    chunks = [indices[i::workers] for i in range(workers)]
    if workers == 1:
        _render_worker((str(film_path), str(out), indices))
        return
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_worker, (str(film_path), str(out), chunk)) for chunk in chunks]
        done = 0
        for fut in cf.as_completed(futs):
            done += fut.result()
            print(f'  rendered {done}/{len(indices)}', flush=True)


def mux_mp4(frames_dir, out_mp4, audio_from):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('ffmpeg not found on PATH')
    audio = Path(audio_from)
    if not audio.is_file():
        raise SystemExit(f'audio source missing: {audio}')
    cmd = [
        ffmpeg, '-y',
        '-framerate', '20', '-i', str(Path(frames_dir) / 'f%04d.png'),
        '-i', str(audio),
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-shortest',
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit('ffmpeg failed:\n' + proc.stderr[-4000:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='out',
                    help='output directory (frames/ and film.mp4)')
    ap.add_argument('--frames', default=None,
                    help='render only A or A..B (default: all 300)')
    ap.add_argument('--no-render', action='store_true', help='only write the .ncel')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args(argv)

    out = Path(os.path.expanduser(args.out))
    film = ROOT / 'ncel/v7/film.ncel'
    print(f'inventory: {len(INV)} keys ({len(USABLE)} used, '
          f'{sum(1 for k in INV if k["id"] >= 100 and k["used"])} extras)')
    text = write_film(film)
    print(f'wrote {film} ({len(text)/1e6:.1f} MB)')
    if args.no_render:
        return

    indices = _parse_frames(args.frames)
    full = indices is None
    if full:
        indices = list(range(300))
    frames_dir = out / 'frames'
    print(f'rendering {len(indices)} frames to {frames_dir} with {args.workers} workers')
    render_frames(film, frames_dir, indices, workers=args.workers)
    if full:
        mux_mp4(frames_dir, out / 'film.mp4', ROOT / 'sources/sound/ink_to_sky_mix.m4a')
        print(f'wrote {out / "film.mp4"}')
    else:
        print('partial range: skipped mp4 mux')


if __name__ == '__main__':
    main()
