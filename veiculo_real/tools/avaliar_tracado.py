#!/usr/bin/env python3
"""Compara metodos de tracado nos ensaios gravados, sem gabarito.

Para cada quadro, o tracado que o metodo estimou (levado ao referencial do mundo
pela odometria do log) tem de PREVER onde estarao os cones PERTO (< 1,6 m, os
mais precisos) vistos nos quadros seguintes, ate 1 s depois: cada cone a +W/2
(borda esquerda) ou -W/2 (direita) da linha central. Mede tambem o salto do
tracado entre quadros seguidos (ja descontado o movimento).

"centro": erro do CENTRO previsto, pela media das duas bordas vistas depois
(mediana das distancias dos cones de cada lado): nao depende da largura, que
muda ao longo da pista; e o que a centralizacao precisa acertar.

    python3 tools/avaliar_tracado.py logs/2026*
    python3 tools/avaliar_tracado.py logs/2026* --mapa             # + SLAM e SVM no lugar
    python3 tools/avaliar_tracado.py logs/2026* --set kappa_noise=0.2

Ensaios de 24/11 (13 voltas): estimador 7,7 cm de erro mediano (p90 24,7);
SLAM + SVM no lugar dele 10,9 cm (p90 45,6).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT/'tools')):
    if path not in sys.path:
        sys.path.insert(0, path)
import replay_corredor as rc  # noqa: E402
from fva_car.corridor import CorridorEstimator, CorridorSettings, sample_centerline  # noqa: E402
from fva_car.lateral import GroundCalibration  # noqa: E402

ARCS = np.arange(0., 3.01, .05)
NEAR_M = 1.6        # so cones perto pontuam
HORIZON_S = (.2, 1.)  # quadros seguintes considerados


def load_frames(folder):
    """Registros do cones_debug.jsonl; linha truncada (volta interrompida) e ignorada."""
    records = []
    with (Path(folder)/'cones_debug.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if 't_capture' in record:
                records.append(record)
    return records


def _compose(pose, motion):
    x, y, th = pose
    dx, dy, dth = motion
    c, s = math.cos(th), math.sin(th)
    return x + c*dx - s*dy, y + s*dx + c*dy, th + dth


def _to_world(points, pose):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    p = np.asarray(points, float).reshape(-1, 2)
    return np.c_[x + c*p[:, 0] - s*p[:, 1], y + s*p[:, 0] + c*p[:, 1]]


def evaluate(folder, calibration, settings, use_map=False):
    """(erros de previsao em m, saltos de e, saltos de psi, fracao de quadros validos,
    erros do centro em m)."""
    estimator = CorridorEstimator(settings, calibration.homography if calibration else None)
    path_map = None
    if use_map:
        from fva_car.map_path import MapPath
        path_map = MapPath(wheelbase=settings.pose_offset_m, replace=True)
    pose, frames = (0., 0., 0.), []
    for record in load_frames(folder):
        left, right, motion, pose_frame = rc.frame_inputs(record)
        moved = motion
        if moved is not None and pose_frame:
            dx, dy, dth = moved
            moved = (dx + settings.pose_offset_m*(math.cos(dth) - 1), dy + settings.pose_offset_m*math.sin(dth), dth)
        if moved is not None and all(np.isfinite(moved)):
            pose = _compose(pose, moved)
        estimate = estimator.step(float(record['t_capture']), left, right, motion, pose_frame)
        if path_map is not None:
            estimate = path_map.step(float(record['t_capture']), left, right, True, motion, pose_frame, estimate,
                                     estimator._point_covariance)
        path = None
        if estimate.valid and estimate.state_vector is not None:
            vector = np.asarray(estimate.state_vector, float)
            path = (_to_world(sample_centerline(vector, ARCS), pose), float(vector[4]))
        frames.append((float(record['t_capture']), pose, left, right, path, estimate.errors if estimate.valid else None,
                       moved))
    errors, center_errors = [], []
    for k, (t, _, _, _, path, _, _) in enumerate(frames):
        if path is None:
            continue
        line, width = path
        tangent = np.gradient(line, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None] + 1e-12
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        for later, pose_later, left, right, _, _, _ in frames[k+1:]:
            if later - t > HORIZON_S[1]:
                break
            if later - t < HORIZON_S[0]:
                continue
            sides = []
            for points, sign in ((left, 1.), (right, -1.)):
                points = np.asarray(points, float).reshape(-1, 2)
                points = points[(points[:, 0] > .2) & (points[:, 0] < NEAR_M)]
                if not len(points):
                    continue
                world = _to_world(points, pose_later)
                nearest = ((world[:, None, :] - line[None])**2).sum(-1).argmin(1)
                inside = (nearest > 2) & (nearest < len(line) - 3)
                if inside.any():
                    offset = np.einsum('nk,nk->n', world[inside] - line[nearest[inside]], normal[nearest[inside]])
                    errors.extend(np.abs(offset - sign*width/2))
                    sides.append(float(np.median(offset)))
            if len(sides) == 2:
                center_errors.append(abs(sum(sides)/2))
    series = [frame[5] for frame in frames]
    jump_e, jump_psi = rc.jumps(series, [frame[6] for frame in frames])
    valid = sum(item is not None for item in series)/max(len(series), 1)
    return np.asarray(errors), jump_e, jump_psi, valid, np.asarray(center_errors)


def report(name, results):
    errors = np.concatenate([r[0] for r in results]) if results else np.empty(0)
    jump_e = np.concatenate([r[1] for r in results]) if results else np.empty(0)
    jump_psi = np.concatenate([r[2] for r in results]) if results else np.empty(0)
    if not len(errors):
        return f'{name}: sem quadros validos'
    center = np.concatenate([r[4] for r in results])
    return (f'{name:28s} previsao dos cones: mediana {100*np.median(errors):4.1f} cm, p90 {100*np.percentile(errors, 90):5.1f} cm'
            f' | centro: mediana {100*np.median(center):4.1f} cm, p90 {100*np.percentile(center, 90):5.1f} cm'
            f' | salto e {100*np.percentile(jump_e, 90):.1f} cm, psi {np.degrees(np.percentile(jump_psi, 90)):.1f} graus (p90)'
            f' | validos {100*np.mean([r[3] for r in results]):.0f}%')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensaios', nargs='+', type=Path)
    parser.add_argument('--calibration', default=str(rc.DEFAULT_CALIBRATION))
    parser.add_argument('--track-width', type=float, default=1.3, help='palpite inicial da largura, m')
    parser.add_argument('--set', action='append', metavar='AJUSTE=VALOR', help='ajuste de CorridorSettings')
    parser.add_argument('--mapa', action='store_true', help='compara tambem o SLAM + SVM no lugar do estimador')
    args = parser.parse_args(argv)
    calibration = GroundCalibration.from_file(args.calibration) if args.calibration else None
    folders = [f for f in args.ensaios if (f/'cones_debug.jsonl').exists()]
    variants = [('estimador', False)] + ([('mapa SLAM + SVM no lugar', True)] if args.mapa else [])
    settings = CorridorSettings.from_value({'track_width_m': args.track_width, 'pose_offset_m': .36,
                                            **rc.parse_set(args.set)})
    for name, use_map in variants:
        print(report(name, [evaluate(folder, calibration, settings, use_map) for folder in folders]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
