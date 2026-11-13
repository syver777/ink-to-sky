"""Raster helpers used by the build.

The grass hill profile is measured by rasterising key 76 (INV index 76)
directly from its mapped vector layers.
"""
import numpy as np
from PIL import Image, ImageDraw


def _fill_polygon(draw, poly, value):
    ext = list(poly.exterior.coords)[:-1]
    draw.polygon([(x, y) for x, y in ext], fill=value)
    for ring in poly.interiors:
        draw.polygon([(x, y) for x, y in list(ring.coords)[:-1]], fill=0)


def rasterize_layers(layers, width=1920, height=1080):
    """Paint a list of (name, colour, geometry) as flat polygons.

    The film only needs the exact fill colours, not antialiasing, so PIL is
    enough and much faster than a full northcel render.
    """
    img = Image.new('RGB', (width, height), (254, 254, 254))
    draw = ImageDraw.Draw(img)
    for _name, colour, geom in layers:
        if geom.is_empty:
            continue
        rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        geoms = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for poly in geoms:
            if poly.geom_type == 'Polygon' and poly.area > 0.2:
                _fill_polygon(draw, poly, rgb)
    return np.asarray(img)


def source_hill(layers, base=740.0, harmonics=3):
    """The grass-scene hill profile, measured from key 76 (v6lib.source_hill)."""
    im = rasterize_layers(layers).astype(np.int16)
    # Any of the palette's grass greens (the source keys use several shades).
    green = ((im[:, :, 1] > 140) & (im[:, :, 0] < 160)
             & (im[:, :, 2] < 130))
    if not green.any():
        return [(float(x), base) for x in range(0, im.shape[1], 8)]
    xs = np.arange(im.shape[1])
    top = np.full(im.shape[1], np.nan)
    for x in range(im.shape[1]):
        ys = np.where(green[:, x])[0]
        ys = ys[(ys > 500) & (ys < 950)]
        if len(ys):
            top[x] = ys.min()
    good = np.isfinite(top)
    if good.sum() < 16:
        return [(float(x), base) for x in range(0, im.shape[1], 8)]
    u = (xs[good] - im.shape[1] / 2) / im.shape[1]
    vals = top[good]
    A = [np.ones_like(u)]
    for k in range(1, harmonics + 1):
        A.append(np.sin(2 * np.pi * k * u))
        A.append(np.cos(2 * np.pi * k * u))
    A = np.stack(A, axis=1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    uu = (xs - im.shape[1] / 2) / im.shape[1]
    B = [np.ones_like(uu)]
    for k in range(1, harmonics + 1):
        B.append(np.sin(2 * np.pi * k * uu))
        B.append(np.cos(2 * np.pi * k * uu))
    B = np.stack(B, axis=1)
    line = B @ coef
    line = line - np.mean(line) + base
    return [(float(x), float(line[x])) for x in range(0, im.shape[1], 8)]
