#!/usr/bin/env python3
"""Replay do estimador de corredor sobre um ensaio gravado (cones_debug.jsonl).

Nao precisa das imagens: usa os cones detectados em cada quadro (em metros),
o lado de cada cor e o deslocamento da odometria registrados no log, e compara
com os erros que o Stanley recebeu na hora.

    python3 tools/replay_corredor.py logs/20261023_080437
    python3 tools/replay_corredor.py logs/a logs/b --csv corredor.csv
    python3 tools/replay_corredor.py logs/a --set kappa_noise=1.0 --set row_sigma=6

Metricas por ensaio:
    quadros validos    antes (log) x estimador
    estados            AMBAS, SO_EXTERNA, ... (fracao dos quadros)
    salto mediano/p90  |diferenca| de e e psi entre quadros validos seguidos,
                       ja descontado o movimento do carro (quanto menor, mais
                       estavel a referencia para o Stanley)
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fva_car.cones import SideTracker  # noqa: E402
from fva_car.corridor import CorridorEstimator, CorridorSettings, STATES  # noqa: E402
from fva_car.lateral import GroundCalibration  # noqa: E402

DEFAULT_CALIBRATION = ROOT/'calibracao'/'camera_lateral.json'


def load_frames(folder):
    path = Path(folder)/'cones_debug.jsonl'
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_inputs(record, sides=None, estimator=None):
    """sides: SideTracker para recalcular o lado de cada cor (codigo atual);
    None usa o lado registrado no log (o que o carro usou)."""
    vision = record.get('vision') or {}
    detection = vision.get('detection') or {}
    green = np.asarray(detection.get('green', {}).get('m') or [], float).reshape(-1, 2)
    red = np.asarray(detection.get('red', {}).get('m') or [], float).reshape(-1, 2)
    logged = ((vision.get('corridor') or {}).get('sides') or {}).get('green_left')
    if logged is None:
        logged = (vision.get('sides') or {}).get('green_left', True)
    green_left = logged
    if sides is not None:
        green_left, _, flipped = sides.update(green, red, estimator, (vision.get('sides') or {}).get('green_left', True))
        if flipped and estimator is not None:
            estimator.reset()
            estimator.learned_width = None
    tracking = vision.get('tracking') or {}
    motion = tracking.get('motion_m')
    pose_frame = tracking.get('motion') == 'odometria'
    left, right = (green, red) if green_left else (red, green)
    return left, right, motion, pose_frame


def _moved(errors, motion):
    """Erros (e, psi) do quadro anterior levados ao quadro atual pelo movimento."""
    if errors is None or motion is None:
        return None
    from fva_car.corridor import offsets
    state = (*errors, 0., 1.)
    d, along, _, _ = offsets([motion[:2]], state)
    psi = errors[1] + errors[2]*float(along[0]) - motion[2]
    return -float(d[0]), math.atan2(math.sin(psi), math.cos(psi))


def jumps(series, motions):
    """|salto| de e e psi entre quadros validos seguidos, descontando o movimento."""
    de, dpsi = [], []
    for k in range(1, len(series)):
        prev, cur = series[k-1], series[k]
        expected = _moved(prev, motions[k])
        if expected is None or cur is None:
            continue
        de.append(abs(cur[0]-expected[0]))
        dpsi.append(abs(math.atan2(math.sin(cur[1]-expected[1]), math.cos(cur[1]-expected[1]))))
    return np.asarray(de), np.asarray(dpsi)


def replay(folder, calibration, settings, writer=None, recompute_sides=False):
    records = load_frames(folder)
    estimator = CorridorEstimator(settings, calibration.homography if calibration else None)
    sides = SideTracker() if recompute_sides else None
    old, new, motions, states = [], [], [], []
    for record in records:
        left, right, motion, pose_frame = frame_inputs(record, sides, estimator)
        estimate = estimator.step(float(record['t_capture']), left, right, motion, pose_frame)
        logged = (record.get('vision') or {}).get('stanley_errors')
        old.append(tuple(logged) if logged else None)
        new.append(estimate.errors)
        states.append(estimate.state)
        if motion is not None and pose_frame and settings.pose_offset_m:
            dx, dy, dth = motion
            motion = (dx + settings.pose_offset_m*(math.cos(dth)-1), dy + settings.pose_offset_m*math.sin(dth), dth)
        motions.append(motion)
        if writer is not None:
            writer.writerow({
                'ensaio': Path(folder).name, 'frame': record['frame'], 't': record['t_capture'],
                'cones_esq': len(left), 'cones_dir': len(right),
                'antes_e': logged[0] if logged else '', 'antes_psi': logged[1] if logged else '',
                'antes_kappa': logged[2] if logged else '',
                'estado': estimate.state, 'valido': int(estimate.valid),
                'e': estimate.errors[0] if estimate.valid else '',
                'psi': estimate.errors[1] if estimate.valid else '',
                'kappa': estimate.errors[2] if estimate.valid else '',
                'largura': round(estimate.width_m, 3) if math.isfinite(estimate.width_m) else '',
                'sigma_e': round(estimate.sigma[0], 4) if math.isfinite(estimate.sigma[0]) else '',
                'aceitos_esq': estimate.used[0], 'aceitos_dir': estimate.used[1],
                'descartados': estimate.rejected})
    return old, new, motions, states


def summary(name, old, new, motions, states):
    n = len(old)
    lines = [f'== {name}: {n} quadros']
    lines.append(f'   validos: antes {sum(e is not None for e in old)}  estimador {sum(e is not None for e in new)}')
    counts = {s: states.count(s) for s in STATES if states.count(s)}
    lines.append('   estados: ' + ', '.join(f'{s} {100*c/n:.0f}%' for s, c in counts.items()))
    for label, series in (('antes', old), ('estimador', new)):
        de, dpsi = jumps(series, motions)
        if len(de):
            lines.append(f'   {label:9s} salto e: mediana {100*np.median(de):5.1f} cm  p90 {100*np.percentile(de, 90):5.1f} cm'
                         f' | psi: mediana {np.degrees(np.median(dpsi)):4.1f} graus  p90 {np.degrees(np.percentile(dpsi, 90)):5.1f}'
                         f'  ({len(de)} pares)')
    kappas = [e[2] for e in new if e is not None]
    if kappas:
        lines.append(f'   curvatura do estimador: p10 {np.percentile(kappas, 10):+.2f}  p90 {np.percentile(kappas, 90):+.2f} 1/m')
    return '\n'.join(lines)


def parse_set(values):
    changes = {}
    for item in values or []:
        key, _, value = item.partition('=')
        changes[key.strip()] = float(value)
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensaios', nargs='+', type=Path)
    parser.add_argument('--calibration', default=str(DEFAULT_CALIBRATION))
    parser.add_argument('--track-width', type=float, default=1.8, help='largura nominal do corredor, m')
    parser.add_argument('--set', action='append', metavar='AJUSTE=VALOR', help='ajuste de CorridorSettings')
    parser.add_argument('--csv', type=Path, help='serie por quadro (antes x estimador)')
    parser.add_argument('--recompute-sides', action='store_true',
                        help='decide o lado de cada cor com o codigo atual (padrao: o lado do log)')
    args = parser.parse_args(argv)
    calibration = GroundCalibration.from_file(args.calibration) if args.calibration else None
    settings = CorridorSettings.from_value({'track_width_m': args.track_width, **parse_set(args.set)})
    handle = writer = None
    if args.csv:
        handle = args.csv.open('w', newline='', encoding='utf-8')
        writer = csv.DictWriter(handle, ['ensaio', 'frame', 't', 'cones_esq', 'cones_dir', 'antes_e', 'antes_psi',
                                         'antes_kappa', 'estado', 'valido', 'e', 'psi', 'kappa', 'largura',
                                         'sigma_e', 'aceitos_esq', 'aceitos_dir', 'descartados'])
        writer.writeheader()
    try:
        for folder in args.ensaios:
            print(summary(folder.name, *replay(folder, calibration, settings, writer, args.recompute_sides)),
                  flush=True)
    finally:
        if handle:
            handle.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
