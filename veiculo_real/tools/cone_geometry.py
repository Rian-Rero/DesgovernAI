"""Centro equidistante das curvas no piso, secoes normais e borda virtual."""

import numpy as np

from cone_tracking import transform_points


def load_calibration(path):
    # Reutiliza a validacao de calibracao existente sem abrir sensores/atuadores.
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from fva_car.lateral import GroundCalibration
    return GroundCalibration.from_file(path)


def image_homography(calibration, shape):
    h, w = shape
    cw, ch = calibration.image_size
    if abs((w/h)/(cw/ch)-1) > .01:
        raise ValueError('Proporcao da imagem diferente da calibracao.')
    scale = np.array([[cw/w, 0., .5*cw/w-.5],
                      [0., ch/h, .5*ch/h-.5], [0., 0., 1.]])
    return calibration.homography @ scale


def smooth_curve(points, count=100):
    """Hermite parametrica por comprimento de arco; aceita curvas em x/y."""
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2:
        return np.empty((0, 2))
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    if len(points) < 2:
        return np.empty((0, 2))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    tangents = np.gradient(points, arc, axis=0)
    rows = np.linspace(0., arc[-1], count)
    indices = np.clip(np.searchsorted(arc, rows, side='right')-1, 0, len(arc)-2)
    h = arc[indices+1]-arc[indices]
    t = ((rows-arc[indices])/h)[:, None]
    return ((2*t**3-3*t**2+1)*points[indices]
            + (t**3-2*t**2+t)*h[:, None]*tangents[indices]
            + (-2*t**3+3*t**2)*points[indices+1]
            + (t**3-t**2)*h[:, None]*tangents[indices+1])


def normals(curve):
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    return np.c_[-tangent[:, 1], tangent[:, 0]]


def normal_intersections(curve, boundary, side):
    """Cruza as normais da curva com segmentos da borda; sem extrapolacao."""
    normal = normals(curve)
    a, d = boundary[:-1], np.diff(boundary, axis=0)
    relative = a[None, :, :]-curve[:, None, :]
    den = d[None, :, 0]*normal[:, None, 1]-d[None, :, 1]*normal[:, None, 0]
    numerator = relative[:, :, 1]*normal[:, None, 0]-relative[:, :, 0]*normal[:, None, 1]
    u = np.divide(numerator, den, out=np.full_like(den, np.nan), where=abs(den) > 1e-9)
    intersection = a[None, :, :]+u[:, :, None]*d[None, :, :]
    distance = np.sum((intersection-curve[:, None, :])*normal[:, None, :], axis=2)
    valid = (u >= 0) & (u <= 1) & (distance*side > 0)
    cost = np.where(valid, abs(distance), np.inf)
    index = np.argmin(cost, axis=1)
    output = intersection[np.arange(len(curve)), index]
    output[~np.isfinite(np.min(cost, axis=1))] = np.nan
    return output


def metric_corridor(left, right, *, margin=.1, min_width=.4, max_width=5., min_radius=.3):
    """Ajusta o centro pelas normais das CURVAS, sem pares de cones.

    O chute inicial usa a faixa longitudinal comum. Apos isso, cada secao
    cruza as duas curvas continuamente e recentra ate convergir.
    """
    left, right = smooth_curve(left), smooth_curve(right)
    if len(left) < 3 or len(right) < 3:
        return None
    start = max(left[:, 0].min(), right[:, 0].min())
    end = min(left[:, 0].max(), right[:, 0].max())
    if end-start < .2:
        return None
    x = np.linspace(start, end, 80)
    lorder, rorder = np.argsort(left[:, 0]), np.argsort(right[:, 0])
    curve = np.c_[x, (np.interp(x, left[lorder, 0], left[lorder, 1])
                     + np.interp(x, right[rorder, 0], right[rorder, 1]))/2]
    for _ in range(4):
        l = normal_intersections(curve, left, 1)
        r = normal_intersections(curve, right, -1)
        valid = np.all(np.isfinite(l), axis=1) & np.all(np.isfinite(r), axis=1)
        # Nao conecta segmentos separados por cruzamentos/ausencia de suporte.
        runs = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1)+1)
        indices = max(runs, key=len)
        if len(indices) < 12:
            return None
        l, r = l[indices], r[indices]
        curve = (l+r)/2
    widths = np.linalg.norm(l-r, axis=1)
    if np.any(widths < max(min_width, 2*margin)) or np.any(widths > max_width):
        return None
    if np.ptp(widths) > max(.3, .4*np.median(widths)):
        return None
    if np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)) < .2:
        return None
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if np.any(np.diff(arc) < 1e-5):
        return None
    tangent = np.gradient(curve, arc, axis=0)
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    curvature = np.gradient(heading, arc)
    if np.any(abs(curvature) > 1/max(min_radius, .01)):
        return None
    return curve, l, r, widths, curvature


def virtual_boundary(visible, width, visible_is_left):
    curve = smooth_curve(visible, 60)
    if len(curve) < 3:
        return np.empty((0, 2))
    return curve + (-1 if visible_is_left else 1)*width*normals(curve)


def project_curve(points, inverse, shape):
    pixels = transform_points(points, inverse)
    h, w = shape
    if (not np.all(np.isfinite(pixels)) or np.any(abs(pixels) > 10000)
            or np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= w)
            or np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= h)):
        raise ValueError('Trajetoria fora da imagem/calibracao.')
    return np.rint(pixels).astype(np.int32)
