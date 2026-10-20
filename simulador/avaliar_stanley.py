"""Avaliacao independente, offline: centro do carro versus textura da pista.

Le a geometria e a textura de /Pista apenas para avaliar logs ja gravados.
Nenhum dado deste modulo e fornecido ao controlador.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def reference_from_shape(data, object_matrix):
    texture = data['texture']
    width, height = texture['resolution']
    rgb = np.frombuffer(texture['texture'], dtype=np.uint8).reshape(height, width, 4)[::-1, :, :3]
    matrix = np.asarray(object_matrix, dtype=float).reshape(3, 4)
    local = np.asarray(data['vertices']).reshape(-1, 3)[data['indices']]
    world = local @ matrix[:, :3].T + matrix[:, 3]
    uv = np.asarray(texture['coordinates']).reshape(-1, 2)
    if np.ptp(world[:, 2]) > 1e-4:
        raise ValueError('A avaliacao requer uma pista plana horizontal.')
    design = np.column_stack((uv, np.ones(len(uv))))
    mapping = np.linalg.lstsq(design, world[:, :2], rcond=None)[0].T
    if np.max(np.abs(design @ mapping.T - world[:, :2])) > 1e-5:
        raise ValueError('As coordenadas UV nao representam uma textura plana unica.')
    pixel_to_uv = np.array([[1/width, 0, .5/width],
                            [0, -1/height, 1-.5/height], [0, 0, 1]])
    pixel_to_world = np.vstack((mapping, [0, 0, 1])) @ pixel_to_uv
    vx, vy = pixel_to_world[:2, 0], pixel_to_world[:2, 1]
    scale = float(np.linalg.norm(vx))
    if not np.isclose(scale, np.linalg.norm(vy), rtol=1e-3) or abs(np.dot(vx, vy)) > scale**2*1e-3:
        raise ValueError('A avaliacao requer pixels isotropicos no plano da pista.')

    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([17, 90, 65], np.uint8), np.array([38, 255, 255], np.uint8))
    # Esqueleto morfologico da textura de referencia (nao da camera de controle).
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = mask.copy()
    while cv2.countNonZero(remaining):
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = cv2.erode(remaining, element)
    if not cv2.countNonZero(skeleton):
        raise ValueError('Nao foi encontrada linha amarela na textura da pista.')
    distances = cv2.distanceTransform(255-skeleton, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * scale
    return rgb, np.linalg.inv(pixel_to_world), distances, scale


def evaluate(logfile, shape_data, object_matrix, settle_time=5):
    folder = Path(logfile)
    data = np.atleast_1d(np.genfromtxt(folder / 'car.csv', delimiter=',', names=True))
    rgb, world_to_pixel, distances, scale = reference_from_shape(shape_data, object_matrix)
    world = np.column_stack((data['x'], data['y'], np.ones(len(data))))
    pixels = world @ world_to_pixel.T
    pixels = (pixels[:, :2] / pixels[:, 2, None]).astype(np.float32)
    errors = cv2.remap(distances, pixels[:, 0].reshape(1, -1), pixels[:, 1].reshape(1, -1),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan')).ravel()
    errors = errors[data['t'] >= settle_time]
    finite = errors[np.isfinite(errors)]
    if not len(finite):
        raise ValueError('Nenhuma amostra valida dentro da textura da pista.')
    metrics = {
        'reference_source': 'yellow centerline skeleton from /Pista texture and UV geometry',
        'vehicle_reference_point': 'body center (Car position), not front axle',
        'reference_pixel_size_m': scale,
        'evaluated_after_s': settle_time,
        'outside_texture_samples': int(np.count_nonzero(~np.isfinite(errors))),
        'cross_track_rms_m': float(np.sqrt(np.mean(finite**2))),
        'cross_track_p95_m': float(np.percentile(finite, 95)),
        'cross_track_peak_m': float(np.max(finite)),
    }
    (folder / 'reference_metrics.json').write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.polylines(overlay, [pixels.astype(np.int32)], False, (255, 80, 0), 3)
    start = tuple(pixels[0].astype(int))
    finish = tuple(pixels[-1].astype(int))
    cv2.circle(overlay, start, 12, (0, 255, 0), -1)
    cv2.circle(overlay, finish, 12, (0, 0, 255), -1)
    cv2.imwrite(str(folder / 'reference_trajectory.png'), overlay)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logfile', help='pasta do ensaio contendo car.csv')
    parser.add_argument('--port', type=int, default=23000)
    args = parser.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sim = RemoteAPIClient(port=args.port).require('sim')
    pista = sim.getObject('/Pista')
    print(json.dumps(evaluate(args.logfile, sim.getShapeViz(pista, 0),
                              sim.getObjectMatrix(pista, -1)), indent=2))
