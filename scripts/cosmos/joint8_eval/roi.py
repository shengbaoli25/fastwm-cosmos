"""Observed-frame spatial attention statistics on GT-only annotated regions."""
import numpy as np


def region_statistics(heat, boxes, view):
    """Fractional token-cell overlap with a union of view-local pixel boxes."""
    heat = np.asarray(heat, dtype=float)
    if heat.shape != (14, 28) or view not in (0, 1):
        raise ValueError('Expected 14x28 dual-view attention grid')
    pixels = np.zeros((224, 224), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
            raise ValueError('ROI outside view')
        pixels[y0:y1, x0:x1] = True
    weight = pixels.reshape(14, 16, 14, 16).mean((1, 3))
    mass = heat[:, view*14:(view+1)*14]
    total, inside, fraction = mass.sum(), (mass*weight).sum(), pixels.mean()
    conditional = inside/total if total > 0 else None
    return dict(view_attention_mass=float(total), region_attention_mass=float(inside),
                area_fraction=float(fraction), within_view_fraction=conditional,
                enrichment=conditional/fraction if conditional is not None and fraction > 0 else None)
